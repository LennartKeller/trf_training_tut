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

# PyTorch Lightning

## Introduction
In contrast to the Huggingface `Trainer`, PyTorch Lightning takes a different approach.
The goal of this framework is to handle the training but also the building of a model itself.
In its philosophy, a model and its inference-, training and prediction logic are no separate things that can be exchanged independently.
Instead, it brings these fuses together by binding all this steps to the model.
In doing so, PyTorch-Lightning does not make many assumptions on the nature of the model or the training itself. Therefore, the user has great flexbility to implement nearly everything that comes into mind.
It provides a custom API that divides the training process in a sequence of single atomic steps.
Also, it offers a lot of valuable tweaks like training with half-precision, automatic tuning of the learning rate, and integrations into hyperparameter tuning frameworks or into command-line interfaces.
The documentation is exhaustive, with many tutorials (texts and videos) and many best practices and user guides on building a model.
It supports different backends to train not only on GPUs but also on other acceleratory like TPUs.
These backends also support distributed training across different machines or processing nodes in a cluster.
Each fixed step or aspect of the training process can be extended or customized by overwriting specific methods or using callbacks.
A PyTorchLightning based project is composed of three main classes. 


## Classes

### Models

__LightningModule__


The neural network (i.e., a lanugage model)  is implemented in a `LightningModule`.
This class inherits from PyTorchs `nn.Module` class. `nn.Modules` are the basic building blocks of neural networks in PyTorch. In essence, they store a set of parameters, for example, weights of a single layer alongside with `.forward`-method that defines the computational logic.
`nn. Modules` are designed to work recursively. One module can be composed of several submodules so that each building block of a neural network, starting from single layers up to a network, can be implemented in this one class.

```{code-cell} ipython3
import torch 
from torch import nn

class DenseLayer(nn.Module):
    """Fully connected linear layer."""
    
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_shape, out_shape), requires_grad=True)
    
    def forward(self, inputs):
        return torch.matmul(inputs, self.weights)

network = nn.Sequential(DenseLayer(512, 16), DenseLayer(16, 8), DenseLayer(8, 2))
inputs = torch.randn(8, 512)  # Batchsize 8
outputs = network(inputs)
print(outputs.size())
print(issubclass(network, nn.Module))
```
The `LightningModule` is intended to replace the outmost `nn.Module` instance, meaning the one that holds a complete network.
It extends the `nn.Module` class with new methods, designed to structure not only the logic of a single forward pass but also other steps like a complete train-, test- or validation-steps.
Similar to plain `nn.Modules`, the `.forward`-methods defines a simple forward step through the network.
The `.train_step` method handles the computation of the loss. Likewise, the `.validation_step` and `.test_step` methods are required to define the calculation of the validation metrics or the prediction of instances.
Logging of the training can also be done directly within these methods, using the polyvalent `.log`- or `log-dict`-methods.

In contrast to plain `PyTorch,` the optimizer is not regarded as an external object. Instead its configuration is moved into the `configure_optimizer`-method of the model.

In `PyTorchLightning` there is a clear differentiation of hyperparameters.
Hyperparameters that affect the model (like the number of hidden layers) are directly assigned to the model and saved with each checkpoint.
By default, each argument of the constructor is considered to be a hyperparameter.
By calling the `.save_hyperparameters`-method in the constructor, these arguments are serialized into a `.hparams`-attribute.
This strategy ensures that while loading an old checkpoint, it is entirely transparent which hyperparameters were used to train it.

For special networks architectures requiring further customization, `LightningModules` also expose lifecycle hooks for many steps throughout the training.

### Data

`PyTorchLightning` also comes with a custom module called `LightningDataModule` that handles data loading, preparation, and splitting.
Its primary purpose is to provide the train-, test- and validation splits of the dataset.
Each split has a corresponding method, which must return a dataloader object.

Operations like loading the data or further preparation (i.e. tokenization) can be implemented by a `.prepare_data`-method.

In a distributed computing environment, additional `.setup`- and `teardown`-methods can be implemented to define operations that should be applied to the data on each computing unit independently.


__Trainer__

The `Trainer` object handles the actual training.
It receives the model and data (wrapped in Lightning modules) alongside all training-specific hyperparameters, like the number of epochs, the devices to train on, or a list of loggers to log the progress.
Customization via subclassing of the trainer is not the intended way. Instead, `PyTorchLightning` provides an API for Plugins and Callbacks that can control the different stages of the training. 

## Features

In addition to the basic features and classes to structure the training, `PyTorchLightning` ships with a set of functions to improve usability or experimental results.

__CLI Interface__

Most notably, it has built-in support for creating command-line interfaces that control all hyperparameters of an experiment.
To do so, it uses Pythons own `argparse` library and each of the three main classes offers methods to add its hyperparameters to the parser. Each class also comes with classmethods, that allow them to be initiated using from parsed arguments.

__Tuning__


## Implemenation






