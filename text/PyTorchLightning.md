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

## Design & Philosophy

PyTorch-Lightning does not make many assumptions and gives (or leaves) many steps into the hand of the user
But it provides the user with a custom API that divides the training process in a sequence of single atomic steps.
It provides a lot of valuable tweaks like training with half-precision, automatic tuning of the learning rate, and integrations into hyperparameter tuning frameworks.
The documentation is exhaustive, with many tutorials (texts and videos) and many best practices and user guides on how to build a model.
It comes with different backends that support multi-device training, not only on GPUs but also on other acceleratory like TPUs. These backends also support distributed training across different machines or processing nodes in a cluster.
Also, it provides ways to control the training via a CLI interface.
Each step or aspect of the training process can be extended or customized by overwriting specific methods or using callbacks.
A PyTorchLightning based project is composed of three main classes. 


## Models

__LightningModule__

* Wraps a model 
* Overwritable methods to control:
	* A single inference step
	* A train step
	* Test/ Val step
	* Configuration of the optimizer
	* Logging
	* Initialization of the model
	* Storing model specific hyperparameters
	* Other hooks for certrain steps of the training process

The model (aka neural network) is implemented in a `LightningModule`.
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

dense = DenseLayer(512, 16)
inputs = torch.randn(8, 512)  # Batchsize 8
outputs = dense(inputs)
print(outputs.size())
```
A `LightningModule` is designed to replace the outmost `nn.Module` instance, meaning the one that holds a complete network.
It extends the `nn.Module` class with new methods, designed to structure not only the logic of a single forward pass but also other steps like a complete train-, test- or validation-steps.
Similar to plain `nn.Modules`, the `.forward`-methods defines a simple forward step through the network.
The `.train_step` method handles the computation of the loss. Likewise, the `.validation_step` and `.test_step` methods are required to define the calculation of the validation metrics or the prediction of instances.
Logging of the training can also be done directly within these methods, using the polyvalent `.log`- or `log-dict`-methods.

In contrast to plain `PyTorch`, where the optimizer is regarded as an external object that can be replaced without changing the model itself, `PyTorchLightning` takes a different approach and moves the optimizer and its configuration into the model via a `configure_optimizer` method.

In `PyTorchLightning` there is a differentiation between hyperparameters that directly belong to the model and those that control different training aspects. This is because all hyperparameters of the model are stored in the model. 
By default, each argument of the constructor is considered to be a hyperparameter.
By calling the `.save_hyperparameters`-method in the constructor, these arguments are saved in a `.hparams`-attribute.
Another way to pass and store hyperparameters to the model is to pass them as dict-like object into the constructor and then call the save
All parameters in this attribute get explicitly saved with each checkpointing of a `LightningModule`.
This strategy ensures that while loading an old checkpoint it is entirly obvious which hyperparameters were used to train it.

For special networks architectures requiring further customization, `LightningModules` also expose lifecycle hooks for many steps throughout the training.


## Data

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






