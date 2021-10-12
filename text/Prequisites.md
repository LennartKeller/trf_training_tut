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

# Prequisites

The following experiments share the same general logic, but the concrete implementation will differ in minor details since each framework has another structural approach.
So before we start, we will take a short look at the general logic for the data loading parts of the experiment, as well as the computation of the loss function and evaluation metrics

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
print("Test")
```

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
print(dataset['train'][0])
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
print(batch)
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

Since stated in the experimental design, we use a plain Mean-Squared-Error regression loss. Still, since we only want to consider the special tokens, we have to select them before the actual computation. To achieve this, we need the `input_ids` to figure out their position in the sequence.









