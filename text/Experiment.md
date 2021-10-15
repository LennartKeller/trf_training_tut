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
pp = pprint.PrettyPrinter(depth=6)
print = pp.pprint
```


# Experimental Design

## Task

```{figure} ./figures/GraphicsTrfTut.png
---
name: fig-task-desc
---
Visualization of the sentence ordering task.
```

To compare the frameworks, we will implement the same experiment with each of them.
The task of the experiment is a critical choice since training a model on a standard task like text- or token classification would not require much customization. Also, it would put the Huggingface Trainer into an advantageous position because it supports such tasks out of the box.
To ensure a fair comparison, we chose another quite exotic objective: Sentence Ordering.
Our goal is to train a model to predict the correct order of a sequence of shuffled sentences.
This task seemed right for two reasons.
Firstly, it can be implemented with a standard Huggingface model but requires a custom loss function.
Secondly, the task falls into the category of self-supervised learning. So it is possible to generate training and test data from unstructured text data in an effortless manner.

But how do we operationalize this task?
There are various methods proposed ranging from relatively simple ones like applying ranking-loss functions (CITE) to rather complex approaches that learn a graph representation of the sentences and then use topological sorting to extract the correct ordering of the sentences (CITE).
Because we do not care much about achieving state-of-the-art results, we opt for one of the most straightforward approaches and frame it as a regression task.
{numref}`fig-task-desc` visualizes this approach.
The model should output a regression score for each sentence in the input, indicating its position in the original text.
Therefore, a special token is added as a prefix to each sentence. This special token is our target while training, and it should output a value near to the original index of the sentence in the correct ordered text.
The loss is measured using the Mean-Squared-Error objective
The target value for each sentence token is not normalized and ranges from 0 to $N$ where $N$ is the number of sentences in the input sequence.
Another even more straightforward approach would be to add a final layer to the network with a fixed size of neurons (one for each sentence), but this would mean we had to know the number of sentences in the input beforehand, which would harm the usability of the model.

Since the position of the target tokens in the input sequences differs, we need our language model to output one logit for each token. Two Huggingface model variants return a suitable output `<...ModelType...>ForTokenClassification` or `<...ModelType...>ForQuestionAnswering`. We chose the first one, but all the code in the following section should also run when employing a model with a question-answering head.

## Metrics

To measure the performance of our model, we use two metrics.

__Accuracy__

Accuracy measures how many sentences per instance are indexed correctly.
Accuracy gives a rough estimate of how well the model performs, but it can paint a misleading picture since it does not fully account for our task's ranking aspect.
For example, suppose that the model would correctly predict that sentence $B$ follows sentence $A$ and expects them to be at position $0$ and $1$ in the total ordering.
But in reality, they are the last sentences of the text. So, in this case, the accuracy would be $0$ (assuming that all other predictions were also wrong).

__Kendalls Tau__

In contrast to accuracy, Kendall Tau is a ranking correlation coefficient that accounts for partially correct parts of a ranking.
It measures the difference between pairs of sentences correctly predicted as following and all other mispredicted pairs. 
To correct for the chance of randomly predicting correct pairs of sentences, the value is divided by the total number of unique ways to pick two sentences from the sequence.

$$
\tau_{\textrm{Kendall}} = \frac{\#\textrm{Correctly predicted pairs of sentences} - \#\textrm{Misredicted pairs of sentences}}{\binom{N}{2}}
$$



## Dataset

We use the ROCStories dataset (version 2017). It consists of 52.665 short stories with a fixed length of five sentences. This dataset is commonly used in the literature because its stories mainly depict concrete actions, making them relatively simple to understand without leaving much space for ambiguities. This property makes it a good fit for testing the general capabilities of language models on this task.

```{warning}
Even though the ROCStories dataset is freely available to the public, anyone who wants to use it has to submit contact data. So the dataset itself is not included in the Github-Repository and must be downloaded independently from https://cs.rochester.edu/nlp/rocstories/
```

```{note}
In addition, we tested the same experimental setup on a dataset of sentences sampled from german short stories (Novellen), but unsurprisingly, that did not work very well.
Applying this task to all kinds of different textual domains can be a fruitful question itself but lies outside the scope of this work.
```

### Dataset-preparation

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
Following the library's out-of-place policy, the `.map`-method returns a new dataset containing the changes instead of changing the original dataset, it was called on.

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