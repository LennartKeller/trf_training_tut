from collections import defaultdict
from typing import Callable, Dict
from dataclasses import dataclass, field

import numpy as np
import torch
from datasets import Dataset
from scipy.stats import kendalltau
from sklearn.metrics import accuracy_score
from torch import nn
from torch._C import EnumType
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    BertConfig,
    BertModel,
    EvalPrediction,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    default_data_collator,
)


@dataclass
class ModelArgs:
    model_name_or_path: str = field(
        default="bert-base-cased",
        metadata={
            "help": "Path to pretrained model or model or id in huggingface hub."
        },
    )

    final_checkpoint_path: str = field(
        default=None, metadata={"help": "Path to save the final model."}
    )


def make_compute_metrics_func(target_token_id) -> Callable:
    def compute_ranking_func(eval_prediction: EvalPrediction) -> Dict[str, float]:
        batch_sent_idx, batch_input_ids = eval_prediction.label_ids
        # We convert the logits with shape (batch_size, seq_len, 1) to be in shape (batch_size, seq_len)
        batch_logits = eval_prediction.predictions.squeeze(2)

        metrics = defaultdict(list)
        for sent_idx, input_ids, logits in zip(
            batch_sent_idx, batch_input_ids, batch_logits
        ):
            sent_idx = sent_idx.reshape(-1)
            input_ids = input_ids.reshape(-1)
            logits = logits.reshape(-1)

            sent_idx = sent_idx[sent_idx != 100]
            target_logits = logits[input_ids == target_token_id]
            if sent_idx.shape[0] > target_logits.shape[0]:
                sent_idx = sent_idx[: target_logits.shape[0]]
            # Calling argsort twice on the logits gives us their ranking in ascending order
            predicted_idx = np.argsort(np.argsort(target_logits))
            tau, pvalue = kendalltau(sent_idx, predicted_idx)
            metrics["kendalls_tau"].append(tau)
            metrics["acc"].append(accuracy_score(sent_idx, predicted_idx))
            metrics["mean_logits"].append(logits.mean())
            metrics["std_logits"].append(logits.std())
        metrics = {metric: np.mean(scores) for metric, scores in metrics.items()}
        return metrics

    return compute_ranking_func


def so_data_collator(batch_entries):
    """
    Custom dataloader to apply padding to the labels.
    TODO document me better :)
    """
    label_dicts = []

    # We split the labels from the rest to process them independently
    for entry in batch_entries:
        label_dict = {}
        for key in list(entry.keys()):
            if "labels" in key:
                label_dict[key] = entry.pop(key)
        label_dicts.append(label_dict)

    # Everything except our labels can easily be handled be transformers default collator
    batch = default_data_collator(batch_entries)

    # We need to pad the labels "manually"
    for label in label_dicts[0]:
        labels = pad_sequence(
            [label_dict[label] for label_dict in label_dicts],
            batch_first=True,
            padding_value=-100,
        )

        batch[label] = labels

    return batch


class SentenceOrderingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.target_token_id = kwargs.pop("target_token_id")
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        # Get sentence indices
        batch_labels = inputs.pop("labels")

        # Get logits from model
        outputs = model(**inputs)
        batch_logits = outputs["logits"]

        # Get logits for all cls tokens
        batch_input_ids = inputs["input_ids"]

        # Since we have varying number of labels per instance, we need to compute the loss manually for each one.
        loss_fn = nn.MSELoss(reduction="sum")
        batch_loss = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)

        for labels, logits, input_ids in zip(
            batch_labels, batch_logits, batch_input_ids
        ):

            # Firstly, we need to convert the sentence indices to regression targets.
            # To avoid exploding gradients, we norm them to be in range 0 <-> 1
            # Also we need to remove the padding entries (-100)
            true_labels = labels[labels != -100].reshape(-1)
            targets = true_labels.float()

            # Secondly, we need to get the logits from each target token in the input sequence
            target_logits = logits[input_ids == self.target_token_id].reshape(-1)

            # Sometimes we will have less target_logits than targets due to trunction of the input
            # In this case, we just consider as many targets as we have logits
            if target_logits.size(0) < targets.size(0):
                targets = targets[: target_logits.size(0)]

            # Finally we compute the loss for the current instance and add it to the batch loss
            batch_loss = batch_loss + loss_fn(targets, target_logits)

        # The final loss is obtained by averaging over the number of instances per batch
        loss = batch_loss / batch_logits.size(0)

        outputs["loss"] = loss
        return (loss, outputs) if return_outputs else loss


def make_tokenization_func(tokenizer, text_column, *args, **kwargs):
    def tokenization(entry):
        return tokenizer(entry[text_column], *args, **kwargs)

    return tokenization


def make_rename_func(mapping, remove_src=False):
    def rename(entry):
        for src, dst in mapping.items():
            if remove_src:
                data = entry.pop(src)
            else:
                data = entry[src]
            entry[dst] = data
        return entry

    return rename
