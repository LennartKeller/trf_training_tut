from pathlib import Path
from re import S
from typing import Any, Dict
from pytorch_lightning.trainer.trainer import Trainer

import torch
from torch import nn
import torchmetrics
from datasets import Dataset
from datasets.load import load_dataset
from pytorch_lightning import Callback, LightningDataModule, LightningModule
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.utils.dummy_pt_objects import AutoModelForSequenceClassification


class HuggingfaceDatasetWrapper(LightningDataModule):

    def __init__(self,
        dataset: Dataset,
        text_column: str,
        target_column: str,
        tokenizer: PreTrainedTokenizerBase,
        train_batch_size: int = 8,
        eval_batch_size: int = 16,
        train_split_name: str = 'train',
        eval_split_name: str ='test',
        test_split_name: str = 'text'
        ):
        super().__init__()
        self.dataset = dataset
        self.text_column = text_column
        self.target_column = target_column
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_split_name = train_split_name
        self.eval_split_name = eval_split_name
        self.test_split_name = test_split_name
    
    def prepare_data(self, tokenizer_kwargs: Dict[str, str] = None):
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {'truncation': True, 'padding':True}
        self.dataset = self.dataset.map(lambda e: self.tokenizer(e[self.text_column], **tokenizer_kwargs), batched=True)
        self.dataset.set_format('torch', columns=['input_ids', 'attention_mask', self.target_column])
        try:
            self.dataset = self.dataset.rename_column(self.target_column, 'labels')
        except ValueError:
            # target column should already have correct name
            pass
    
    def train_dataloader(self):
        return DataLoader(self.dataset[self.train_split_name], batch_size=self.train_batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.dataset[self.eval_split_name], batch_size=self.eval_batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.dataset[self.test_split_name], batch_size=self.eval_batch_size)




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
        loss = outputs['loss']
        if loss is None:
            raise Exception('No loss returned by model. Check your input data.')
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        outputs = self.forward(batch)
        loss = outputs['loss']
        if loss is None:
            raise Exception('No loss returned by model. Check your input data.')
        self.log('val_loss', loss, prog_bar=True, logger=True)
        # accuracy
        labels = batch['labels']
        predictions = outputs.logits.argmax(dim=1)
        acc = torchmetrics.functional.accuracy(target=labels, preds=predictions)
        self.log('val_acc', acc, prog_bar=True, logger=True)
    
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
        loss = outputs['loss']
        if loss is None:
            raise Exception('No loss returned by model. Check your input data.')
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.eval()
        outputs = self.forward(batch)
        loss = outputs['loss']
        if loss is None:
            raise Exception('No loss returned by model. Check your input data.')
        self.log('val_loss', loss, prog_bar=True, logger=True)
        # accuracy
        labels = batch['labels']
        predictions = outputs.logits.argmax(dim=1)
        acc = torchmetrics.functional.accuracy(target=labels, preds=predictions)
        self.log('val_acc', acc, prog_bar=True, logger=True)
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return opt
        
    

class PlAutoModelForSequenceClassification(PlTransformerBaseModel):

    def __init__(self, hparams):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(hparams.model_name_or_path)
        self.save_hyperparameters()
        







# TODO Remove
class HuggingfaceTrainer(Trainer):

    def save_checkpoint(self, filepath, weights_only: bool = False) -> None:
        pl_model = self.model
        transformer = getattr(pl_model, 'model', None)
        if transformer is None:
            raise Exception("Could not find model")
        transformer.save_pretrained(filepath)
        super().save_checkpoint(filepath=filepath, weights_only=weights_only)
    


class SaveHuggingfaceModelCheckpointCallback(Callback):

    def __init__(self, dir, steps=None) -> None:
        super().__init__()
        self.dir = Path(dir)
        if self.dir.is_file():
            raise Exception('Save dir should be directory but is a file')
        self.steps = steps
    
    def _save_model(self, pl_module, checkpoint_name):
        model = pl_module.model
        model.save_pretrained(self.dir / checkpoint_name)

    def on_batch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.steps is None:
            return
        if trainer.global_step % self.steps == 0:
            self._save_model(pl_module=pl_module, checkpoint_name=f'checkpoint-{trainer.global_step}')
            
    
    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if self.steps is not None:
            return
        