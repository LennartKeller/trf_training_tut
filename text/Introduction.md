# Introduction

Transformer-based neural language models have revolutionized the world of NLP.
Their ability to process long-range dependencies within texts and gain language processing abilities via self-supervised pretraining allowed them to set-state-of the art results across many different tasks.
Thus they are of great interest to a growing community of people across numerous scientific disciplines and the industry.
The most popular source of (pretrained) language models is Huggingface, a startup that provides various models alongside additional tooling.
While most of the time it is not necessary to train a language model from scratch, it has to be finetuned to the desired task, or it has to be adapted to a specific domain.
Even though finetuning is less computationally expensive than training a model from scratch, it remains challenging.
Requirements may include multi-gpu training, data loading, ensuring reproducibility, hyperparameter organization, tracking or early stopping.
There are numerous frameworks to facilitate this process. This work aims to compare a selection of these frameworks and test their suitability to use them with the Huggingface infrastructure.

The rest of this work is structured as follows: First, the Huggingface Transformers library, the core of the Huggingface ecosystem, is shortly presented. Afterwards, the selected training frameworks are presented.
The actual comparison by carrying out the same experiment with each of them. Finally, a conclusion is drawn.



