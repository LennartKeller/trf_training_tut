# Experimental Design

## Task

```{figure} ./figures/GraphicsTrfTut.png
---
name: fig-task-desc
---
Visualization of the sentence ordering task.
```

Various frameworks aim at streamlining the training of neural networks for the user.
We chose th
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

We use the ROCStories dataset (version 2017). It contains 52.665 short stories with a fixed length of five sentences. This dataset is commonly used in the literature because its stories mainly depict concrete actions, making them relatively simple to understand without leaving much space for ambiguities. This property makes it a good fit for testing the general capabilities of language models on this task.

```{warning}
Even though the ROCStories dataset is freely available to the public, anyone who wants to use it has to submit contact data. So the dataset itself is not included in the Github-Repository and must be downloaded independently from https://cs.rochester.edu/nlp/rocstories/
```

```{note}
In addition, we tested the same experimental setup on a dataset of sentences sampled from german short stories (Novellen), but unsurprisingly, that did not work very well.
Applying this task to all kinds of different textual domains can be a fruitful question itself but lies outside the scope of this work.
```