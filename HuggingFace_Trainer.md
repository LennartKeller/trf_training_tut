# Huggingface Trainer

## Facts

* Designed to work with Huggingface Transfomers Models
	* Since these inherit from nn.Modules these might work as well
* Provide training on multiple GPUs and with half precision
* Logging and checkpointing also implemented

## Design

* The Trainer itself is an object
* Arguments are passed as an TrainingArguments object
* The model and the tokenizer are passed alongside with the TrainingArguments
	to the trainer.
* The Training process is structured in different methods of the trainer:
	* To customize the training each method can be overwritten
	*
* Training Loop can be extended via Callbacks (TrainerCallbacks)
* A list of callables can be passed to compute metrics
* Datasets are passed as plain torch.Datasets

## First impression

* Undocumented: You have to look into the code quite often
* Extending every step during training with overwritting corresponding methods seems nice.
* Test if this works with custom models.
