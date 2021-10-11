from typing import Any, Callable, Dict, List, Tuple, Union
import torch
from torch import FloatStorage, nn
from torch.nn.utils.rnn import pad_sequence
from transformers import default_data_collator

####################################################
#                                                  #
# TODO to avoid duplications move me to own module #
#                                                  #
####################################################


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

    # We need to pad the labels 'manually'
    for label in label_dicts[0]:
        labels = pad_sequence(
            [label_dict[label] for label_dict in label_dicts],
            batch_first=True,
            padding_value=-100,
        )

        batch[label] = labels
    return batch


class PoutyneSequenceOrderingLoss:

    def __init__(self, target_token_id):
        self.target_token_id = target_token_id
    
    def __call__(self, outputs, targets) -> float:
        batch_labels = targets["labels"]
        batch_logits = outputs["logits"]
        batch_input_ids = targets["input_ids"]

        # Since we have varying number of labels per instance, we need to compute the loss manually for each one.
        loss_fn = nn.MSELoss(reduction="sum")
        batch_loss = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        for labels, logits, input_ids in zip(batch_labels, batch_logits, batch_input_ids):

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

        return loss


def hugginface_loss(outputs: Dict[str, Any], targets: Any) -> float:
    """
    Returns the loss of the huggingface transformers model.
    """
    return outputs["loss"]


def dummy_loss(*args, **kwargs):
    print(args)
    print(kwargs)
    return 1.0


class TransformerPoutyneCollator:
    def __init__(
        self, y_keys: Union[str, List[str]] = None, custom_collator: Callable = None
    ):
        self.y_keys = y_keys
        self.custom_collator = (
            custom_collator if custom_collator is not None else default_data_collator
        )

    def __call__(self, inputs: Tuple[Dict]) -> Tuple[Dict, Any]:
        batch_size = len(inputs)
        batch = self.custom_collator(inputs)
        if self.y_keys is None:
            y = torch.tensor(float("nan")).repeat(batch_size)
        elif isinstance(self.y_keys, list):
            # If we want to compute the loss later on we can remove the labels from input since we do not need the original loss.
            y = {
                key: batch.pop(key) if "labels" in key else batch.get(key)
                for key in self.y_keys
            }
        else:
            y = batch.get(self.y_keys)

        return batch, y


class TransformerPoutyneWrapper(nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.transformer)})'
    def forward(self, inputs):
        return self.transformer(**inputs)
