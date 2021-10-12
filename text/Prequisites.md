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

# Prequisites

The following experiments share the same structural logic, but the concrete implementation will differ in minor details since each framework has another structural approach.
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
from pprint import pprint
from datasets import Dataset

dataset = Dataset.from_csv('../scripts/data/rocstories')
pprint(dataset.features)
```

```{code-cell} ipython3
pprint(dataset[0])
```

To feed the data to the neural network, we have to split it up into batches of a fixed size.
To do so, PyTorch provides a general class, called `torch.utils.data.DataLoader`, that takes in iterable and returns batches just in time while training. To know

