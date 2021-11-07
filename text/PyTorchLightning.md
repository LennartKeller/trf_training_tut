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

In contrast to the Huggingface `Trainer`, which handles the complete training itself, PyTorch Lightning takes a different structure approach.
The goal of this framework is to handle the training but also the building of a model itself.
In its philosophy, a model and its inference-, training and prediction logic are no separate things that can be exchanged independently.
Instead, it brings these fuses together by binding all these steps to the model itself.
In doing so, PyTorch-Lightning does not make many assumptions on the nature of the model or the training itself and allows flexibility. However, the user has to implement these steps again manually.
Like the Huggingface `Trainer`, Pytorch Lightning provides an API composed of three different main classes, dividing the training process in a sequence of single atomic steps.

It has a quite steep learning curve, especially for beginners, because it requires writing a lot of logic manually.
But it provides exhaustive documentation with many tutorials (texts and videos), best practices, and user guides on building various models across different domains.
On a more pro-level, it offers a lot of valuable tweaks like training with half-precision, automatic tuning of the learning rate, and integrations into hyperparameter tuning frameworks or creating command-line interfaces.
Also, it supports different backends to train not only on GPUs but also on other accelerators like TPUs.
Some of these backends even support distributed training across different machines or processing nodes in a cluster.


## Classes

As stated before, a PyTorchLightning based project comprises three main classes that implement the model, the storing and processing of the training data and the training process itself.
### `LightningModule`

The neural network (i.e., a language model)  is implemented using the `LightningModule`.
This class is an extended version of PyTorch's `nn.Module` class.

`nn.Modules` are the basic building blocks of neural networks in PyTorch. In essence, they store a set of parameters, for example, weights of a single layer alongside with `.forward`-method that defines the computational logic.
`nn. Modules` are designed to work recursively. One module can be composed of several submodules so that each building block of a neural network, starting from single layers up to a network, can be implemented in this one class.
[Listing](markdown-fig) shows an exemplary implementation of a densely-connected, feed-forward layer as `nn.Module`.

```{code-cell} ipython3 markdown-fig
import torch 
from torch import nn

class DenseLayer(nn.Module):
    """Fully connected linear layer."""
    
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_shape, out_shape), requires_grad=True)
    
    def forward(self, inputs):
        return torch.matmul(inputs, self.weights)

network = nn.Sequential(DenseLayer(512, 16), nn.ReLU(), DenseLayer(16, 8), nn.ReLU(), DenseLayer(8, 2))
inputs = torch.randn(8, 512)  # Batchsize 8
outputs = network(inputs)
print(outputs.size())
print(issubclass(nn.Sequential, nn.Module))
```

By wrapping multiple instances of this layer into a `nn.Sequential` object, we can create a simple feed-forward network. As shown at the end of the listing, the complete network again is a subclass of the `nn.Module` class.

A `LightningModule` is intended to replace the outmost `nn.Module` instance of a model, meaning the one that holds a complete network.
It extends the `nn.Module` class with new methods, designed to structure not only the logic of a single forward pass but also other steps like a complete train-, test- or validation-steps.
Like plain `nn.Modules`, the `.forward`-methods defines a simple forward step through the network.

### Training, Testing, Validation

Additionally, there  `.train_step`, `.validation_step` and `.test_step` methods that define how the network is trained, validated and tested.
The `.train_step` is intended to define the computation of the loss for a single batch of data. The other two methods for testing and validation define how the model is evaluated.
### Optimizer

<!--In contrast to plain `PyTorch,` the optimizer is not regarded as an external object. Instead, it is directly bound to a model and its configuration is moved into the `configure_optimizer`-method of the model.-->

### Logging

Since the logic of training and validation is bound to a model, logging must also be implemented here.
A `LighntingModule` has to possible options for logging metrics. A `.log`-methods that can log a single score or a `log-dict`-methods that can be used to log multiple scores stored in a dictionary (with names as keys and the score as values).
These methods can be used from any method making them not only feasible for logging validation but also training scores or other things.

### Model Hyperparameters

In `PyTorchLightning` there is a clear differentiation of hyperparameters.
For example, Hyperparameters that affect the model (like the number of hidden layers) are directly assigned to the model and saved with each checkpoint.
By default, each argument of the constructor of a `LighntningModule` is considered to be a hyperparameter.
By calling the `.save_hyperparameters`-method in the constructor, these arguments are serialized into a `.hparams`-attribute.
This strategy ensures that while loading an old checkpoint, it is entirely transparent which hyperparameters were used to train it.

```{code-cell} ipython3
from pytorch_lightning import LightningModule

class LightningNetwork(LightningModule):


```

### `LighningDataModule`

`PyTorchLightning` also comes with a custom solution for data operations called `LighningDataModule`.
Like the `LightningModule`, it holds the code to load and prepare the data for training and testing.
A class derived from `LighntningDataModule` must implement four required methods. 
The `.prepare_data`-method should implement all steps required to load the data and convert it in a suitable format.
To return the splits for training, testing and evaluation, there are `.train|.test|.val_loader`-methods. Each of them has to return a `DataLoader` object.

Another feature of the `LightningDataModule` is its ability to adapt to distributed environments.
While the `.prepare_data`-method is called once at the beginning of the training, there are also additional `.setup`- and `teardown`-methods.
These methods can define operations that are executed on each computing unit independently. 
While the `.setup`-method is called before the training and the `.teardown`-method is executed after the training is finished.

### `Trainer`

The `Trainer` object handles the actual training.
It receives the model and data (wrapped in Lightning modules) alongside all training-specific hyperparameters, like the number of epochs, the devices to train on, or a list of loggers to log the progress.
Customization via subclassing of the trainer is not the intended way. Instead, `PyTorchLightning` provides an API for Plugins and Callbacks that can control the different stages of the training.
Callbacks are intended to implement steps that are not strictly necessary for training. Instead, they can define additional steps, like logging or applying non-essential operations to the model (i.e., weigh pruning after each epoch).
Like the other main modules, the `Trainer` itself stores all parameters relevant to the training.



## Additional features

### CLI Interface

Most notably, it has built-in support for creating command-line interfaces that control all hyperparameters of an experiment.
It provides the `LightninArgumentParser`, which can be used to parse possible arguments from the constructors of `Lightning'classes.

```python
:name: lightning-parser-listing
parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
parser.add_class_arguments(TensorBoardLogger, nested_key="tensorboard")
parser.add_lightning_class_args(Trainer, "trainer")
parser = PlLanguageModelForSequenceOrdering.add_model_specific_args(parser)
```

### Tuning

In addition to the basic features and classes to structure the training, `PyTorchLightning` ships with a set of functions to improve usability or experimental results.
