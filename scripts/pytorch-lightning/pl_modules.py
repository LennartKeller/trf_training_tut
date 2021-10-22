from collections import defaultdict
from os import stat
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torchmetrics
from datasets import Dataset
from pytorch_lightning import Callback, LightningDataModule, LightningModule
from pytorch_lightning.trainer.trainer import Trainer
from scipy.stats.stats import kendalltau
from sklearn.metrics import accuracy_score
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    PreTrainedTokenizerBase,
    default_data_collator,
)


# TODO to avoid duplications move me to own module
def so_data_collator(batch_entries, label_key="so_targets"):
    """
    Custom dataloader to apply padding to the labels.
    TODO document me better :)
    """
    label_dicts = []

    # We split the labels from the rest to process them independently
    for entry in batch_entries:
        label_dict = {}
        for key in list(entry.keys()):
            if label_key in key:
                labels = entry.pop(key)
                if not isinstance(labels, torch.Tensor):
                    if isinstance(labels, np.ndarray):
                        labels = torch.from_numpy(labels)
                    if isinstance(labels, list):
                        labels = torch.tensor(labels)
                    else:
                        raise Exception(
                            "Labels should be of type torch.Tensor, np.ndarray or list"
                        )
                label_dict[key] = labels
        label_dicts.append(label_dict)

    # Everything except our labels can easily be handled be transformers default collator
    batch = default_data_collator(batch_entries)

    # We need to pad the labels 'manually'
    for label in label_dicts[0]:
        labels = pad_sequence(
            [label_dict[label] for label_dict in label_dicts],
            batch_first=True,
            padding_value=-100,
        )
        batch[label] = labels
    return batch


class HuggingfaceDatasetWrapper(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        text_column: str,
        target_column: str,
        tokenizer: PreTrainedTokenizerBase,
        train_batch_size: int = 8,
        eval_batch_size: int = 16,
        mapping_funcs: List[Callable] = None,
        collate_fn: Callable = default_data_collator,
        train_split_name: str = "train",
        eval_split_name: str = "val",
        test_split_name: str = "test",
    ):
        super().__init__()
        self.dataset = dataset
        self.text_column = text_column
        self.target_column = target_column
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.mapping_funcs = mapping_funcs
        self.collate_fn = collate_fn
        self.train_split_name = train_split_name
        self.eval_split_name = eval_split_name
        self.test_split_name = test_split_name

    def prepare_data(self, tokenizer_kwargs: Dict[str, str] = None):
        # 1. Apply user defined preparation functions
        if self.mapping_funcs:
            for mapping_func in self.mapping_funcs:
                dataset = dataset.map(mapping_func, batched=True)

        # 2. Tokenize the text
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {
                "truncation": True,
                "padding": "max_length",
                "add_special_tokens": False,
            }
        self.dataset = self.dataset.map(
            lambda e: self.tokenizer(e[self.text_column], **tokenizer_kwargs),
            batched=True,
        )
        # 3. Set format of important columns to torch
        self.dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", self.target_column]
        )
        # 4. If the target columns is not named 'labels' change that
        try:
            self.dataset = self.dataset.rename_column(self.target_column, "labels")
        except ValueError:
            # target column should already have correct name
            pass

    def train_dataloader(self):
        return DataLoader(
            self.dataset[self.train_split_name],
            batch_size=self.train_batch_size,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset[self.eval_split_name],
            batch_size=self.eval_batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset[self.test_split_name],
            batch_size=self.eval_batch_size,
            collate_fn=self.collate_fn,
        )

    def map(self, *args, **kwargs):
        self.dataset = self.dataset.map(*args, **kwargs)


class PlLanguageModelForSequenceOrdering(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.base_model = AutoModelForTokenClassification.from_pretrained(
            self.hparams["model_name_or_path"],
            return_dict=True,
            output_hidden_states=True,
            num_labels=1,
        )

    def forward(self, inputs: Dict[Any, Any]) -> Dict[Any, Any]:
        # We do not want to compute token classificaiton loss so we remove the labels temporarily

        labels = inputs.pop("labels")
        outputs = self.base_model(**inputs)

        # # Compute logits for each token in the input squence
        # last_hidden_state = outputs['last_hidden_state']
        # logits = self.linear(last_hidden_state)
        # outputs['logits'] = logits

        # And reattach them later on ...
        inputs["labels"] = labels
        return outputs

    def _compute_loss(self, batch_labels, batch_logits, batch_input_ids) -> float:
        # Since we have varying number of labels per instance, we need to compute the loss manually for each one.
        loss_fn = nn.MSELoss(reduction="sum")
        batch_loss = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        for labels, logits, input_ids in zip(
            batch_labels, batch_logits, batch_input_ids
        ):

            # Firstly, we need to convert the sentence indices to regression targets.
            # To avoid exploding gradients, we norm them to be in range 0 <-> 1
            # labels = labels / labels.max()
            # Also we need to remove the padding entries (-100)
            true_labels = labels[labels != -100].reshape(-1)
            targets = true_labels.float()

            # Secondly, we need to get the logits from each target token in the input sequence
            target_logits = logits[
                input_ids == self.hparams["target_token_id"]
            ].reshape(-1)

            # Sometimes we will have less target_logits than targets due to trunction of the input
            # In this case, we just consider as many targets as we have logits
            if target_logits.size(0) < targets.size(0):
                targets = targets[: target_logits.size(0)]

            # Finally we compute the loss for the current instance and add it to the batch loss
            batch_loss = batch_loss + loss_fn(targets, target_logits)

        # The final loss is obtained by averaging over the number of instances per batch
        loss = batch_loss / batch_logits.size(0)

        return loss

    def _forward_with_loss(self, inputs):
        outputs = self(inputs)

        # Get sentence indices
        batch_labels = inputs["labels"]
        # Get logits from model
        batch_logits = outputs["logits"]
        # Get logits for all cls tokens
        batch_input_ids = inputs["input_ids"]

        loss = self._compute_loss(
            batch_labels=batch_labels,
            batch_logits=batch_logits,
            batch_input_ids=batch_input_ids,
        )
        outputs["loss"] = loss

        return outputs

    def training_step(self, inputs: Dict[Any, Any], batch_idx: int) -> float:
        outputs = self._forward_with_loss(inputs)
        loss = outputs["loss"]
        self.log("loss", loss, logger=True)
        return loss

    def validation_step(self, inputs, batch_idx):
        outputs = self._forward_with_loss(inputs)

        # Detach all torch.tensors and convert them to np.arrays
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = value.detach().cpu().numpy()
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.detach().cpu().numpy()

        # Get sentence indices
        batch_labels = inputs["labels"]
        # Get logits from model
        batch_logits = outputs["logits"]
        # Get logits for all cls tokens
        batch_input_ids = inputs["input_ids"]

        metrics = defaultdict(list)
        for sent_idx, input_ids, logits in zip(
            batch_labels, batch_input_ids, batch_logits
        ):
            sent_idx = sent_idx.reshape(-1)
            input_ids = input_ids.reshape(-1)
            logits = logits.reshape(-1)

            sent_idx = sent_idx[sent_idx != 100]
            target_logits = logits[input_ids == self.hparams["target_token_id"]]
            if sent_idx.shape[0] > target_logits.shape[0]:
                sent_idx = sent_idx[: target_logits.shape[0]]

            # Calling argsort twice on the logits gives us their ranking in ascending order
            predicted_idx = np.argsort(np.argsort(target_logits))
            tau, pvalue = kendalltau(sent_idx, predicted_idx)
            acc = accuracy_score(sent_idx, predicted_idx)
            metrics["kendalls_tau"].append(tau)
            metrics["acc"].append(acc)
            metrics["mean_logits"].append(logits.mean().item())
            metrics["std_logits"].append(logits.std().item())

        metrics["loss"] = outputs["loss"].item()

        # Add val prefix to each metric name and compute mean over the batch.
        metrics = {
            f"val_{metric}": np.mean(scores).item()
            for metric, scores in metrics.items()
        }
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)
        return metrics

    def test_step(self, inputs, batch_idx):
        return self.validation_step(inputs, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams["lr"])

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("PlLanguageModelForSequenceOrdering")
        parser.add_argument(
            "--model.model_name_or_path", type=str, default="bert-base-cased"
        )
        parser.add_argument("--model.lr", type=float, default=3e-5)
        parser.add_argument("--model.target_token_id", type=int, default=101)
        return parent_parser


class HugginfaceWrapper(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        self.train()
        outputs = self.forward(batch)
        loss = outputs["loss"]
        if loss is None:
            raise Exception("No loss returned by model. Check your input data.")
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        outputs = self.forward(batch)
        loss = outputs["loss"]
        if loss is None:
            raise Exception("No loss returned by model. Check your input data.")
        self.log("val_loss", loss, prog_bar=True, logger=True)
        # accuracy
        labels = batch["labels"]
        predictions = outputs.logits.argmax(dim=1)
        acc = torchmetrics.functional.accuracy(target=labels, preds=predictions)
        self.log("val_acc", acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
        return opt


class PlTransformerBaseModel(LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        outputs = self.model(**inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        self.train()
        outputs = self.forward(batch)
        loss = outputs["loss"]
        if loss is None:
            raise Exception("No loss returned by model. Check your input data.")
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        outputs = self.forward(batch)
        loss = outputs["loss"]
        if loss is None:
            raise Exception("No loss returned by model. Check your input data.")
        self.log("val_loss", loss, prog_bar=True, logger=True)
        # accuracy
        labels = batch["labels"]
        predictions = outputs.logits.argmax(dim=1)
        acc = torchmetrics.functional.accuracy(target=labels, preds=predictions)
        self.log("val_acc", acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return opt


class PlAutoModelForSequenceClassification(PlTransformerBaseModel):
    def __init__(self, hparams):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hparams.model_name_or_path
        )
        self.save_hyperparameters()


class SaveHuggingfaceModelCheckpointCallback(Callback):
    def __init__(self, dir, steps=None) -> None:
        super().__init__()
        self.dir = Path(dir)
        if self.dir.is_file():
            raise Exception("Save dir should be directory but is a file")
        self.steps = steps

    def _save_model(self, pl_module, checkpoint_name):
        model = pl_module.model
        model.save_pretrained(self.dir / checkpoint_name)

    def on_batch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.steps is None:
            return
        if trainer.global_step % self.steps == 0:
            self._save_model(
                pl_module=pl_module, checkpoint_name=f"checkpoint-{trainer.global_step}"
            )

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.steps is not None:
            return
