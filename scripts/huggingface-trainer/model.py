import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, BertModel, BertConfig


class BertForMultiHeadConfig(PretrainedConfig):
    
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
            print(losses)
            loss = torch.stack(losses).sum(dim=-1)
            base_outputs['loss'] = loss
            
        return base_outputs
    
    def _init_weights(self, module):
        return self.bert._init_weights(module)