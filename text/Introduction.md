# Introduction

Transformer-based neural language models have revolutionized the world of NLP.
Their ability to process long-range dependencies within texts and gain language processing abilities via self-supervised pretraining allowed them to set-state-of the art results across many different tasks.
But training a transformer, like any other neural network, can be challenging because it may require custom tweaks or steps to make it successful and reproducible.
Depending on the scale of the network these additional steps include things like multi-GPU training, data loading, hyperparameter search, progress tracking, or early stopping.
There are numerous frameworks to facilitate this process.
This work aims to compare a selection of these frameworks and test their suitability to use them with the most popular source of language models: The Huggingface transoformers ecosystem.
To do so, we follow a two-step procedure.
At first, we look at the design and philosphy of each framework to see how it structures the typical deep learning workflow. Then we carry out an experiment with it, to see if theres a gap between the world of theoretical features and reality.

The rest of the work is structured as followed: Firstly a short introduction into the huggingface infrastructure and the special requirements when working with language models are given.
After that, the task for practical evaluation is presented. Finally, in the central part of this work, we theoretically evaluate each framework and use it to employ our experiment.
Finally, we conclude.

<!--Thus they are of great interest to a growing community of people across numerous scientific disciplines and the industry.
The most popular source of (pretrained) language models is Huggingface, a startup that provides various models alongside additional tooling.
While most of the time it is not necessary to train a language model from scratch, it has to be finetuned to the desired task, or it has to be adapted to a specific domain.
Even though finetuning is less computationally expensive than training a model from scratch, it remains challenging.-->