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
from pprint import pprint
from datasets import Dataset, DatasetDict
```

At first, we load the dataset in its original format.

```{code-cell} ipython3
dataset = Dataset.from_csv('/mnt/data/users/keller/ROCStories/ROCStories_winter2017 - ROCStories_winter2017.csv')
dataset
```

We got 52.665 stories. Each one has a length of five sentences. Also we have short title for each story, but we discard them and won't include it in our shuffled texts.

```{code-cell} ipython3
len(dataset)
```

```{code-cell} ipython3
pprint(dataset[0])
```

As next step, we prepare the text by shuffling the sentences and creating labels for each entry indicating the original order. Also we add special tokens as prefix for each sentence.

To apply this steps to the data, we use the `.map`-method of the `Dataset`-class. Like nearly all other methods of this class it works out-of-place, meaning that it returns a new dataset with the changes instead of changing the datset it was called on.

The `.map`-method works in two modes: bacht or non-batched. In either mode it receives a dictionary as input where each key represents a column if the dataset.
In non-bachted mode the values of the input-dictionary are the values of one entry in the dataset. In batched mode they are lists with multiple entries.
The following funcation only works in both modes since it converts both input formats to the same intermediate format, but in terms of speed batched-mode should be preffered.

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
            text = f'{sep_token} ' + f' {sep_token} '.join([s[0] for s in sents_with_idx]) 
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

`[CLS]` is the special token of BERT directly derived models, which while pretraining learns a represenation of the whole input sequence. We will use it as our sentence-token.

```{code-cell} ipython3
map_func = make_shuffle_func('[CLS]')
```

```{code-cell} ipython3
dataset = dataset.map(map_func, batched=True)
```

Now our dataset has two additional columns: The `text`column contains the shuffled and concatenated sentences and the `so_targets` columnm contains the indicies of the sentences in the original order. For example in the first text in the dataset the first sentence in the shuffled text is at the 4th place in the original order.

```{code-cell} ipython3
pprint(dataset[0])
```

Lastly we want to split our dataset into the subset: The train-set will be used for training, the validation set can be used to validate the performance while the training or for hyperparamter optimazation and the testset will be used for the final evulation of a trained and optimized model.

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
