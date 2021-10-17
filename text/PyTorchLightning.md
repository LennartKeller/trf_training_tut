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

```python
import torch 
from torch import nn

class DenseLayer(nn.Module):

	def __init__(self, in_shape, out_shape):
		self.weights = torch.randn(in_shape, out_shape)
	
	def foward(self, inputs):
		return torch.matmul()
```
A `LightningModule` is designed to replace the outmost `nn.Module` instance, meaning the one that holds a complete network.
It extends the `nn.Module` class with new methods, designed to structure not only the logic of a single forward pass but also other steps like a complete train-, test- or validation-steps.
Similar to plain `nn.Modules` the `.forward`-methods defines a simple foward step through the network. The `.train_step` method handles the computation of the loss, while the `.validation_step` and `.test_step` methods are required to define how the metrics and inference is handled.
Logging of the training can also be done directly within these methods, using several different `.log`-methods.

In contrast to plain `PyTorch` where the optimizer is regarded as an external object that can be replaced without changing the model itself, `PyTorchLightning` takes a different approach and moves the optimizer and its configuration into the model via a `configure_optimizer` method.

In `PyTorchLightning` there is a differantiation of hyperparameters that directly belong the model itself and other ones that controll different aspects of the training. All hyperparameters of the model are stores in the model itself. They are defined while the initialization of the object in the constructor and can be safed using a `.save_hyperparameters`-method that should be called at the end of initialization. This method takes all attributes that are not 'nn.Modules' (e.g. do not directly belong the model) and stores them in an `.hparams` atttribute. All parameters in this attribute get explicitly saved when checkpointing a `LightningModule`. This behaviour ensures that while loading an old checkpoint it is entirly obvious which hyperparemters were used to train it.

For special networks architectures that require further customization `LightningModules` also expose lifecycle hooks for many steps throughout the the training like at be beginning or end of one epoch.


## Data

`PyTorchLightning` also comes with a custom module that stores the data for training.
It has three main methods wich return dataloader objects for the train-, test- and validation set.
Also it has `.prepare_data`-method that handled the all preperation steps. Similiar to the `LighntingModule` it also stores all hyperparameters that are relevant to the data.
Because one focus of `PyTorchLightning` is the simplicifaction of training in a distributed setting, additional `.setup`- and `teardown`-methods can be implemented to handle operations to prepare data on each computing unit.
  
## Training

__Trainer__

The training itself is hanlded by a `Trainer` object. It receives the model and data (both wrapped in their Lightning modules) and all training specific hyperparameters, like the number of epochs, the devices to train on, or a list of loggers to log the progress. The trainer is not really intended to be customized via subclassing it. Instead `PyTorchLightning` provides an API for callbacks that can controll nearly every step of the training.

## Features

