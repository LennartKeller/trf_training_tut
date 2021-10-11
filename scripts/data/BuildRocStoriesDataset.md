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
from datasets import Dataset, DatasetDict
```

```{code-cell} ipython3
dataset = Dataset.from_csv('/mnt/data/users/keller/ROCStories/ROCStories_winter2017 - ROCStories_winter2017.csv')
dataset
```

```{code-cell} ipython3
len(dataset)
```

```{code-cell} ipython3
from random import shuffle
from random import seed as set_seed

def make_shuffle_func(sep_token):
    def shuffle_stories(entries, seed=42):
        set_seed(seed)
        entries_as_dicts = [dict(zip(entries, values)) for values in zip(*entries.values())]
        converted_entries = []
        for entry in entries_as_dicts:
            sents = [entry[key] for key in sorted([key for key in entry.keys() if key.startswith('sentence')], key=lambda x: int(x[-1]))]
            sent_idx = list(range(len(sents)))
            sents_with_idx = list(zip(sents, sent_idx))
            shuffle(sents_with_idx)
            text = f'{sep_token} ' + f' {sep_token} '.join([s[0] for s in sents_with_idx]) 
            so_targets = [s[1] for s in sents_with_idx]
            shuffled_entry = {'text': text, 'so_targets': so_targets}
            converted_entries.append(shuffled_entry)
        new_entry = {key: [entry[key] for entry in converted_entries] for key in converted_entries[0]}
        return new_entry
    return shuffle_stories
```

```{code-cell} ipython3
map_func = make_shuffle_func('[CLS]')
```

```{code-cell} ipython3
dataset = dataset.map(map_func, batched=True)
```

```{code-cell} ipython3
dataset
```

```{code-cell} ipython3
train_test = dataset.train_test_split(test_size=0.2, seed=42)

test_validation = train_test['test'].train_test_split(test_size=0.3, seed=42)

dataset = DatasetDict({
    'train': train_test['train'],
    'test': test_validation['train'],
    'val': test_validation['test']})
```

```{code-cell} ipython3
dataset.save_to_disk('rocstories')
dataset
```

```{code-cell} ipython3
for entry in dataset['train']:
    print(entry['text'])
    break
```

```{code-cell} ipython3

```
