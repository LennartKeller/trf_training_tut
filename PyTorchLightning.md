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
	* "Detect bottlenecks"
	* Checkpoint the model during training
	* ArgumentParsing from command line

* Integrations into other PyTorch ecosystems packages like RayTune exist
* Documentation is solid and additionally features BestPractices, Boilerplates
		and Video tutorials
* Open Question how to integrate the tokenizer?
* How is data in general handled?
