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

## Data

__LightningDataModule__

Handles data-preparation:

* Overwritable methods to control:
	* prepare_data: Tokenization...
	* setup / teardown: Prepare things on all accelerators in distributed mode
	* (train|val|test)_dataloader: Return dataset splits (as torch dataloader objects)

  
## Training

__Trainer__

Similar to Huggingface Trainer.

The model and the data (both as lightning modules) is passed to the trainer. The trainer itself stores  training specific hyperparameters (n_epochs, n_val_step) and can be extendend with callbacks. 
Also, loggingg