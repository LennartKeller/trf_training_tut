# PyTorch Lightning

# Design & Philosophy

* Each model is wrapped into a PyTorch Lightning Module
* These modules have methods for each part of the trainning
* Training is done the a Trainer object
* Support for multi GPU training and half precision training
* Many additional features to:
	* Find Optimal batch size
	* Find optimal learning rate
	* Visualize training
	* "Detect bottlenecks."
	* Checkpoint the model during training
	* ArgumentParsing from command line

* Integrations into other PyTorch ecosystems packages like RayTune exist
* Documentation is solid and additionally features BestPractices, Boilerplates
		and Video tutorials
* Open Question how to integrate the tokenizer?
* How is data in general handled?

#### Main Modules

##### Data

__LightningDataModule__

Handles Data preperation and splitting

Possible operations:

* prepare_data: Tokenization...
* setup / teardown: Prepare things on all accelerators in distributed mode
* (train|val|test)_dataloader: Return dataset splits (as torch dataloader objects)

__Plan__
We use to do the preparation and tokenization stuff

##### Models

___LightningModule__

Stores a module and defines all training and/ or inference logic.
Extension of torchs nn.Module class

Code is organized in 5 sections:
* Initializations
* Training Loop
* Validation Loop
* Test Loop
* Optimizer configuration (=> just initiated and return the optimizer)
  
##### Training

__Trainer__

Similar to Huggingface Trainer,
=> Handles the training of the model

Can be extended via custom callbacks (subclassed from pytorch lightning callback class)

The Trainer also handles multi device training on accelerators of varying types.

Functions of Trainer:
(train|eval|test)_loop
checkpointing
compute_metrics
checkpointig
logging
device dispatching
seeding for reproducibility
tweaks to optimize results like gradient clipping, finding the right learning rate, mixed precision, tracking gradients etc
profiling

##### TODOs

* Check if loading from checkpoint restores the correct weights
* Check if logging can be improved (e.g. are the val metrics that are logged averaged over the complete val set?)
* check what callbacks are available (=> How can we use tensorboard for tracking the experiment)
* Does multi gpu training work out of the box?
* Are there some cool features that can improve training