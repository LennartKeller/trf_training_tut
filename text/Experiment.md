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

To read in the data and shuffle the sentences as well as adding the special tokens we use Huggingface Datasets library which provides a variety of useful functions for manipulating text data.
Because a Huggingface Datasets are fully compatible with PyTorchs own class for data loading it can also be used by all non-Huggingface libraries without any further adjustments.

