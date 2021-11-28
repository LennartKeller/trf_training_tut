---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: ["remove-cell"]

from datasets import set_caching_enabled
set_caching_enabled(False)

import pprint
pp = pprint.PrettyPrinter(depth=6, compact=True)
print = pp.pprint
```

# Prerequisites

The following experiments share the same general logic, but the concrete implementation will differ in minor details since each framework has another structural approach.
So before we start, we will take a short look at the general logic for the data loading parts of the experiment, as well as the computation of the loss function and evaluation metrics

## Dataset-preparation

To load the stories, shuffle the sentences, and further prepare, we use Huggingface's Datasets library, which provides various useful functions for manipulating text data.
Because Huggingface Datasets are fully compatible with PyTorch's class for data-loading, they can also be used by all non-Huggingface libraries without further adjustments.

The preparation itself is simple:

```{code-cell} ipython3
from datasets import Dataset, DatasetDict
```

At first, we load the dataset in its original format.

```{code-cell} ipython3
dataset = Dataset.from_csv('../scripts/data/ROCStories_winter2017 - ROCStories_winter2017.csv')
dataset
```

We got 52.665 stories. Each one has a length of five sentences. Additionally, each text has a short title, but we discard them.

```{code-cell} ipython3
len(dataset)
```

```{code-cell} ipython3
print(dataset[0])
```

Next, we create the training data by shuffling the sentences and creating labels indicating the original order. Also, we add special tokens to each sentence.

We implement the shuffling process using the `.map`-method of the `Dataset`-class.
Following the library's out-of-place policy, the `.map`-method returns a new dataset containing the changes instead of changing the dataset it was called on.

The `.map`-method has two modes: batch-mode or single entry mode. In either way it receives a dictionary as input where each key represents a column of the dataset.
In single entry mode, the values of the input dictionary hold one entry in the dataset.
In batch mode, the values are lists containing more than one entry.
The following function only works in both modes since it converts both input formats to the same intermediate form, but in general, the batch mode should be preferred to save time.
The output of the function has to be a dictionary in the same format as the input.

```{code-cell} ipython3
from random import shuffle
from random import seed as set_seed

def make_shuffle_func(sep_token):
    def shuffle_stories(entries, seed=42):
        set_seed(seed)
        entries_as_dicts = [
            dict(zip(entries, values))
            for values in zip(*entries.values())
        ]
        converted_entries = []
        for entry in entries_as_dicts:
            sents = [
                entry[key]
                for key in sorted(
                    [key for key in entry.keys() if key.startswith('sentence')
                    ], key=lambda x: int(x[-1])
                )
            ]
            sent_idx = list(range(len(sents)))
            sents_with_idx = list(zip(sents, sent_idx))
            shuffle(sents_with_idx)
            text = f'{sep_token} ' + f' {sep_token} '.join(
                [s[0]for s in sents_with_idx]
            ) 
            so_targets = [s[1] for s in sents_with_idx]
            shuffled_entry = {'text': text, 'so_targets': so_targets}
            converted_entries.append(shuffled_entry)
        new_entry = {
            key: [entry[key] for entry in converted_entries]
            for key in converted_entries[0]
        }
        return new_entry
    return shuffle_stories
```

`[CLS]` is one of the specials tokens of models directly descending from BERT. During the pretraining stage, it learns a representation of the whole input sequence and thus only occurs once in each input.
Since we do not need a representation of the input as a whole, we use it as the special sentence token.

```{code-cell} ipython3
map_func = make_shuffle_func('[CLS]')
```

```{code-cell} ipython3
dataset = dataset.map(map_func, batched=True)
```

After applying the shuffle function, the dataset has two additional columns. The `text` column contains the shuffled and concatenated sentences, and the `so_targets` column contains the indices of the sentences in the original order. For example, in the first text in the dataset, the first sentence in the shuffled text is 4th place in the original order.

```{code-cell} ipython3
print(dataset[0])
```

Lastly, we want to split our dataset into three subsets.
The train-set is used for training.
The validation set can be used to validate the performance during training or hyperparameter optimization.
The test set will be used for the final evaluation of the final model.

```{code-cell} ipython3
train_test = dataset.train_test_split(test_size=0.2, seed=42)

test_validation = train_test['test'].train_test_split(test_size=0.3, seed=42)

dataset = DatasetDict({
    'train': train_test['train'],
    'test': test_validation['train'],
    'val': test_validation['test']})
dataset
```

Finally, we save the dataset.

```{code-cell} ipython3
dataset.save_to_disk('rocstories')
```


```{code-cell} ipython3
:tags: ["remove-cell"]
! rm -r rocstories
```

## Data loading

```{figure} ./figures/DataFlow.png
---
name: fig-dataflow
---
High level visualization of dataflow while training a neural network.
```

As stated before, we will use the same Dataset in the Huggingface format for each run

From a high-level view, a Huggingface `Dataset` can be seen as a table with columns that correspond to attributes (called features) and rows representing one dataset entry.
In a more concrete technical perspective, the `Dataset`-instance provides an iterable that yields a dictionary for each entry in the Dataset. Each dictionary contains attribute-value pairs.

```{code-cell} ipython3
from datasets import load_from_disk

dataset = load_from_disk('../scripts/data/rocstories')
print(dataset['train'].features)
```

```{code-cell} ipython3
print(dataset['train'][0])
```

We can't feed the model with raw texts, so we have to tokenize them before hand.

As stated before each models comes with a custom tokenizer, so we have to load it, just like the model itself.

```{code-cell} ipython3
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased', return_dict=True)
tokenized_text = tokenizer("Jimmy went down the road.")
print(tokenized_text)
```

The tokenizer takes a text or a collection of texts and converts it to a tokenized sequence. Also it creates additional inputs for the model such as the attention mask.

To tokenize the whole dataset, we can once again use the map function.

```{code-cell} ipython3

def make_tokenization_func(tokenizer, text_column, *args, **kwargs):
    def tokenization(entry):
        return tokenizer(entry[text_column], *args, **kwargs)
    return tokenization

tokenization = make_tokenization_func(
    tokenizer=tokenizer,
    text_column="text",
    padding="max_length",
    truncation=True,
    add_special_tokens=False,
    return_tensors='np'
)

dataset = dataset.map(tokenization, batched=True)
print(dataset['train'][0].keys())
```

To feed the data to the neural network, we have to split it up into batches of a fixed size.
To do so, PyTorch provides a general class, called `torch.utils.data.DataLoader`, that takes in iterable and returns batches just in time while training. 

The `DataLoader` class is agnostic towards the data it receives. To create batches that are compatible with the Huggingface model, we have to pass it a function that takes in multiple entries from our dataset and converts them into the correct format.

This function is called `collate_fn` and can be specified while initiating the `DataLoader` object.
Using a simple identity function, we see that the `collate_fn` receives a tuple with $B$ entries where $B$ is the batch size.

```{code-cell} ipython3
from torch.utils.data import DataLoader

def identity(batch):
    return batch

data_loader = DataLoader(dataset['train'], batch_size=2, collate_fn=identity)
batch = next(iter(data_loader))
print(len(batch))
print(type(batch))
print(batch[0].keys())
```

Huggingface provides a collate function that can convert tokenized data into batches in a suitable format.
The Huggingface collation function only works with numeric data such as scalars or arrays. So we have to drop all texts before we pass the dataset into the dataloader object.

```{code-cell} ipython3
dataset = dataset.remove_columns(
    ["text", "storyid", "storytitle"] + [f"sentence{i}" for i in range(1, 6)]
)
dataset.set_format("torch")
print(dataset["train"].features)
```

After only numeric data is left, we have to face the last problem in the collation problem.
The Huggingface collation function only handles arrays of the same shape when collating them into one batch. In theory (e.g. with other datasets), we could have a varying number of labels, if we wanted to work shuffled texts with a variable number of sentences. We tackle this problem by introducing a custom collation function to make our preparation pipeline as flexible as possible.

```{code-cell} ipython3
from transformers import default_data_collator
from torch.nn.utils.rnn import pad_sequence

def so_data_collator(batch_entries, label_key='so_targets'):
    """
    Custom dataloader to apply padding to the labels.
    """
    label_dicts = []

    # We split the labels from the rest to process them independently
    for entry in batch_entries:
        label_dict = {}
        for key in list(entry.keys()):
            if label_key in key:
                label_dict[key] = entry.pop(key)
        label_dicts.append(label_dict)

    # Everything except our labels can easily be handled by the "default collator"
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
```
This function used the Huggingface default collation function to handle everything except the labels. The labels are padded with a batch-wise max length strategy and added to the batch.

```{code-cell} ipython3
data_loader = DataLoader(dataset['train'], batch_size=2, collate_fn=so_data_collator)
batch = next(iter(data_loader))
print(batch)
```
Now the data is in the correct format for training.

## Loss function

As stated in the experimental design, we use a plain Mean-Squared-Error regression loss. Still, we only want to consider the special tokens, so we must select them before the actual computation. Therefore, we need the `input_ids` to figure out their position in the sequence.
To compute the loss for one single batch, we add the loss scores of all sentences of one text and take the average of all batch entries.
Due to computational constraints, transformer-based language models typically have a limit on the input size. So our inputs might have to be truncated to fit into the model. In this case, we discard the labels for the sentences left out and only consider the data that fits into the model.

The following listing contains a general implementation of the loss function:

```python
import torch
from torch import nn

def sentence_ordering_loss(batch_logits, batch_targets, batch_input_ids) -> torch.Tensor:
    # Since we have varying number of labels per instance, we need to compute the loss manually for each one.
    loss_fn = nn.MSELoss(reduction="sum")
    batch_loss = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    for labels, logits, input_ids in zip(
        batch_labels, batch_logits, batch_input_ids
    ):
        # Firstly, we need to convert the sentence indices to regression targets.
        # Also we need to remove the padding entries (-100)
        true_labels = labels[labels != -100].reshape(-1)
        targets = true_labels.float()

        # Secondly, we need to get the logits from each target token in the input sequence
        target_logits = logits[input_ids == self.target_token_id].reshape(-1)

        # Sometimes, we will have less target_logits than targets due to trunction of the input.
        # In this case, we just consider as many targets as we have logits
        if target_logits.size(0) < targets.size(0):
            targets = targets[: target_logits.size(0)]

        # Finally we compute the loss for the current instance and add it to the batch loss
        batch_loss = batch_loss + loss_fn(targets, target_logits)

    # The final loss is obtained by averaging over the number of instances per batch
    loss = batch_loss / batch_logits.size(0)

    return loss
```
