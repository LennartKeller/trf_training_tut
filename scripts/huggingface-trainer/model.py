from typing import Callable, Dict
import torch
from datasets import Dataset
from torch import nn
from torch._C import EnumType
from torch.nn.utils.rnn import pad_sequence
from transformers import (BertConfig, BertModel, PretrainedConfig,
                          PreTrainedModel, Trainer, default_data_collator)
from transformers import EvalPrediction
from collections import defaultdict
from scipy.stats import kendalltau
import numpy as np

def make_compute_metrics_func(target_token_id) -> Callable:
    def compute_ranking_func(eval_prediction: EvalPrediction) -> Dict[str, float]:
        batch_sent_idx, batch_input_ids = eval_prediction.label_ids
        # We convert the logits with shape (batch_size, seq_len, 1) to be in shape (batch_size, seq_len)
        batch_logits = eval_prediction.predictions.squeeze(2)
        
        metrics = defaultdict(list)
        for sent_idx, input_ids, logits in zip(batch_sent_idx, batch_input_ids, batch_logits):
            sent_idx = sent_idx[sent_idx != 100].reshape(-1)
            target_logits = logits[input_ids == target_token_id]
            if sent_idx.shape[0] > target_logits.shape[0]:
                sent_idx = sent_idx[:target_logits.shape[0]]
            metrics['kendalls_tau'] = kendalltau(sent_idx, target_logits.reshape(-1).argsort())
        
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
            if 'labels' in key:
                label_dict[key] = entry.pop(key)
        label_dicts.append(label_dict)
    
    # Everything except our labels can easily be handled be transformers default collator
    batch = default_data_collator(batch_entries)
    
    # We need to pad the labels "manually"
    for label in label_dicts[0]:
        labels = pad_sequence(
            [label_dict[label] for label_dict in label_dicts],
            batch_first=True,
            padding_value=-100)

        batch[label] = labels

    return batch


class SentenceOrderingTrainer(Trainer):


    def __init__(self, *args, **kwargs):
        self.target_token_id = kwargs.pop('target_token_id')
        super().__init__(*args, **kwargs)


    def compute_loss(self, model, inputs, return_outputs=False):
        
        # Get sentence indices
        batch_labels = inputs.pop('labels')
        
        # Get logits from model
        outputs = model(**inputs)
        batch_logits = outputs['logits']

        # Get logits for all cls tokens
        batch_input_ids = inputs['input_ids']
        
        # Since we have varying number of labels per instance, we need to compute the loss manually for each one.
        loss_fn = nn.MSELoss(reduction='sum')
        batch_loss = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        for labels, logits, input_ids in zip(batch_labels, batch_logits, batch_input_ids):
            
            # Firstly, we need to convert the sentence indices to regression targets.
            # To avoid exploding gradients, we norm them to be in range 0 <-> 1
            # Also we need to remove the padding entries (-100)
            true_labels = labels[labels != -100].reshape(-1)
            #targets = true_labels / true_labels.max()
            targets = true_labels.float()

            # Secondly, we need to get the logits from each target token in the input sequence
            target_logits = logits[input_ids == self.target_token_id].reshape(-1)
            #target_logits = torch.nn.functional.sigmoid(target_logits)

            # Sometimes we will have less target_logits than targets due to trunction of the input
            # In this case, we just consider as many targets as we have logits
            if target_logits.size(0) < targets.size(0):
                targets = targets[:target_logits.size(0)]
            
            # Finally we compute the loss for the current instance and add it to the batch loss
            batch_loss = batch_loss + loss_fn(targets, target_logits)
            #print('Begin')
            #print('Out: ', target_logits)
            #print('Target: ', targets)
            
        
        # The final loss is obtained by averaging over the number of instances per batch
        loss = batch_loss / batch_logits.size(0)
        outputs['loss'] = loss
        return (loss, outputs) if return_outputs else loss 












# MultiHeadTraining (remove asap)
""" class BertForMultiHead(PretrainedConfig):
    
    valid_kwargs = set(['num_multi_labels'])
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        
        return_unused_kwargs = kwargs.get('return_unused_kwargs', False)
        if not return_unused_kwargs:
            kwargs['return_unused_kwargs'] = True
            
        config, unused_kwargs = BertConfig.from_pretrained(*args, **kwargs)
    
        additional_kwargs = {key: value for key, value in unused_kwargs.items() if key in cls.valid_kwargs}
        
        for kwarg, value in additional_kwargs.items():
            setattr(config, kwarg, value)
        
        if return_unused_kwargs:
            unused_kwargs = {key: value for key, value in unused_kwargs.items() if key not in cls.valid_kwargs}
            return config, unused_kwargs
        else:
            return config
    


class BertForMultiHeadModel(PreTrainedModel):
    
    config_class = BertForMultiHeadConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        
        
        num_classes = getattr(config, 'num_labels', None)
        num_labels = getattr(config, 'num_multi_labels', None)
    

        self.classification_head = nn.Linear(in_features=config.hidden_size, out_features=num_classes)
        self.multilabel_classification_head = nn.Linear(in_features=config.hidden_size, out_features=num_labels)
        self.regression_head = nn.Linear(in_features=config.hidden_size, out_features=1)
        self.init_weights()
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                multi_labels=None,
                regression_targets=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None
               ):
        
        classification_labels = labels
        multilabel_classification_labels = multi_labels
        
        return_dict = True
        base_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
            )
        
        pooled_representation = base_outputs.last_hidden_state.mean(dim=1) # cls
        
        classification_logits = self.classification_head(pooled_representation)
        multilabels_logits = self.multilabel_classification_head(pooled_representation)
        regression_logits = self.regression_head(pooled_representation)
        
        base_outputs['classification_logits'] = classification_logits
        base_outputs['multilabels_logits'] = multilabels_logits
        base_outputs['regression_logits'] = regression_logits
        
        losses = []
        if classification_labels is not None:
            #print(classification_logits.size())
            predictions = classification_logits.view(-1, self.config.num_labels)
            #print(predictions.size())
            targets = classification_labels.view(-1)
            #print(predictions.argmax(dim=1))
            loss_func = nn.CrossEntropyLoss()
            classification_loss = loss_func(predictions, targets)
            base_outputs['classification_loss'] = classification_loss
            losses.append(classification_loss)
        
        if multilabel_classification_labels is not None:
            #print(multilabels_logits.size())
            predictions = multilabels_logits.view(-1, self.config.num_multi_labels)
            #print(predictions)
            #print(predictions.size())
            targets = multilabel_classification_labels.view(-1, self.config.num_multi_labels).float()
            #print(predictions.topk(k=3).indices)
            #print(targets)
            loss_func = nn.BCEWithLogitsLoss()
            multilabel_classification_loss = loss_func(predictions, targets)
            base_outputs['multilabel_classification_loss'] = classification_loss
            losses.append(multilabel_classification_loss)
        
        if regression_targets is not None:
            predictions = regression_logits.view(-1)
            #print(predictions)
            targets = regression_targets.view(-1)
            #print(predictions.size())
            loss_func = nn.MSELoss()
            regression_loss = loss_func(predictions, targets)
            base_outputs['regression_loss'] = regression_loss
            losses.append(regression_loss)
        
        if losses:
            #print(losses)
            loss = torch.stack(losses).sum(dim=-1)
            base_outputs['loss'] = loss
            
        return base_outputs
    
    def _init_weights(self, module):
        return self.bert._init_weights(module) """
