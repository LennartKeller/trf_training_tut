# Introduction

Transformer-based neural language models have revolutionized the world of NLP.
Their ability to process long-range dependencies within texts and gain language processing abilities via self-supervised pretraining allowed them to set-state-of the art results across many different tasks.
These successes made them a technique with great interest across many disciplines of academia and the industry alike.

But training a transformer can be challenging.
One reason for this is the state of the software landscape.
The high rate of innovations in this field drives the development of new models and architectures.
Software libraries trying to keep up with this rapid pace have to regularly adapt to new models and techniques, complicating building robust and stable software.
The other reason lies in the complexity of the models.
Like any other neural network, training a transformer bases mode is a complicated process that requires a lot of technical and domain knowledge.
Also, a robust understanding of how neural networks work and what pitfalls must be avoided is needed.
While no software can replace domain knowledge and general understanding, numerous frameworks aim at lowering the barrier of entry by mitigating the technical complexities.
Depending on the scale of the network, these technical hurdles may include things like multi-GPU training, data loading, hyperparameter search, progress tracking, early stopping, and many more.
The goal of this work is two-fold: Firstly, the goal is to compare three of these frameworks by using them to train a language model on predicting the correct order of a sequence of shuffled sentences.
Furthermore, this work can also be seen as a tutorial that guides through the basic features of each framework and shows how to adapt them to work with language models.
As the underlying source of pretrained language models, the `transformers` library from Huggingface, which has become the de-facto standard for these models, is used.

The following frameworks are chosen: The built-in `Trainer` of the `transformers` library itself, PyTorch Lightning, a framework, which offers a general framework to train neural networks of all kinds; and Poutyne, a library which tries to bring the ease of Tensorflow's Keras API to PyTorch.
This selection is intended to segment the broader set of available frameworks by their complexity. Du its origin, the `Trainer` is highly optimized for working with `transformers`. Thus it can be seen as the middle ground between the pro-level solution PyTorch Lightning and the more beginner-friendly Poutyne framework.

The rest of the works is structured as follows:
Firstly, a brief introduction to the Huggingface software stack, which serves as the foundation for all experiments, is given.
Next, the experimental design is presented. Since the frameworks only control the actual training of the models, many parts of the experiment will be mostly the same.
These steps are laid out in the next chapter to avoid redundancies in the following chapters, where each framework is presented, and the implementational details are discussed.
Then, the results of all experiments are analyzed to check if the choice of a framework affects the model's performance. And, of course, to see if the sentence ordering works in general.
Finally, we end with a concluding comparison of the frameworks carving out their strengths and weaknesses and discussing their different potential for specific use cases.
