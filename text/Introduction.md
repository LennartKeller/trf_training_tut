# Introduction

Transformer-based neural language models have revolutionized the world of NLP.
Their ability to process long-range dependencies within texts and gain language processing abilities via self-supervised pretraining allowed them to set-state-of the art results across many different tasks.
But training a transformer, like any other neural network, can be challenging.
Obtaining the best results possible may require custom tweaks or steps.
Depending on the scale of the network, these additional steps include things like multi-GPU training, data loading, hyperparameter search, progress tracking, or early stopping.
There are numerous frameworks to facilitate this process.
This work aims to compare a selection of these frameworks and test their suitability when combined with models from the Huggingface library.
To do so, we follow a two-step procedure.
At first, we look at the design and philosophy of each framework to see how it structures the typical deep learning workflow. Then we experiment with it to see if there is a gap between the world of theoretical features and reality.

The rest of the work is structured as follows: Firstly, a short introduction to the specialties of language models and the Huggingface ecosystem is given.
After that, the task for practical evaluation is presented. Finally, in the central part of this work, we theoretically evaluate each framework and use it to employ our experiment.
Finally, we conclude.