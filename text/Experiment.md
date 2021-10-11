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
# Experimental Design

## Task

```{figure} ./figures/GraphicsTrfTut.png
---
name: fig-task-desc
---
Visualization of the sentence ordering task.
```

To compare the frameworks, we will implement the same experiment with each of them.
The task of the experiment is a critical choice, since training a model on a standard task like text- or token classification could be implemented without much customization. Also, it would put the Huggingface Trainer into an advantageous position because it supports standard tasks out of the box.
To ensure a fair comparison, we chose another quite exotic task: Sentence Ordering.
Our goal is to train a model to predict the right order of a sequence of shuffled sentences.
This task seemed right for two reasons. The first being that it can be implemented using a standard Huggingface model with a custom loss function.
Secondly, the task falls into the category of self-supervised learning. So it is possible to generate training and test data from unstructured text data in an effortless manner.
But how do we operationalize this task?
There are various methods proposed ranging from relatively simple ones like applying ranking-loss functions (CITE) to rather complex approaches that learn a graph representation of the sentences and then use topological sorting to extract the correct ordering of the sentences (CITE).
Because we do not care much about achieving state-of-the-art results, we opt for one of the most straightforward approaches and frame it as a regression task.
So we want our model to output a regression score for each sentence in the input, indicating its position in the original text.
{numref}`fig-task-desc` visualizes this approach.
We add a special token to each sentence which is used as our target token to evaluate against. As loss function we chose a simple MSE regression loss.
Another even more straightforward approach would be to add a final layer to the network with a fixed size of neurons (one for each sentence), but this would mean we had to know the number of sentences in the input beforehand, which would harm the usability of the model.
So our model needs to output one logit for each token it receives. There are two possible Huggingface model variations, which can be used for this: `<...ModelType...>ForTokenClassification` or `<...ModelType...>ForQuestionAnswering`. We chose the first one, but all the code in the following section should also run when employing a question answering model.

## Dataset

We use the ROCStories dataset (version 2017). It consits of short stories with a fixed length of five sentences. This dataset is commonly used in the literature because its stories are relatively simple to understand, not leaving much space for ambiguities. This property makes it a good fit for testing the general capabilities of language models on this task.

```{note}
Eventhough, the ROCStories dataset is freely available to the public, anyone who wants to use it has to submit contact data. So the dataset itself is not included in the Github-Repository and must be downloaded independently from https://cs.rochester.edu/nlp/rocstories/
```

In addition, we used the same experimental setup on a dataset of sentences sampled from german short stories (Novellen), but unsurprisingly, that didn't work very well.
Applying this task to all kinds of different textual domains can be a fruitful question itself but lies outside the scope of this work.

## Data-preparation

To load the stories, shuffle the sentences, and further prepare, we use Huggingface Datasets library, which provides various useful functions for manipulating text data.
Because Huggingface Datasets are fully compatible with PyTorch's class for data-loading, it can also be used by all non-Huggingface libraries without any further adjustments.

The prepreparation itself is simple:
<!-- Include building dataset notebook here...-->
```{code-cell} ipython3
from pprint import pprint
from datasets import Dataset, DatasetDict
```

At first, we load the dataset in its original format.

```{code-cell} ipython3
dataset = Dataset.from_csv('../scripts/data/ROCStories_winter2017 - ROCStories_winter2017.csv')
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


```{code-cell} ipython3
{
    "tags": [
        "remove-cell"
    ]
}
! rm -r rocstories
```