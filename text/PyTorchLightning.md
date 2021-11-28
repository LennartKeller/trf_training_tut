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

In contrast to the Huggingface `Trainer`, which handles the complete training itself, PyTorch Lightning ({cite:t}`falcon2019pytorchlightning`) takes a different approach.
It not only aims at handling the training but also at structuring the creation of a model too.
Its main goal is not to hide complexity from the user but to provide a well structured API to building a deep learning models.
The most striking aspect of this is that in PyTorch Lightning's philosophy, a model and its inference-, training and prediction logic are not separate things that can be exchanged independently.
Instead, it binds them directly to the model itself.
In doing so, PyTorch-Lightning does not make any assumptions on the nature of the model or the training itself. Thus it allows covering many tasks and domains with maximum flexibility.

However, this approach comes at the cost that the user again must implement many things manually.
Naturaly, this approach is keener to researchers who implement and test custom models, while practitioners who only want to employ pre-built models must deal with some implementational overhead.
PyTorch Lightning's steep learning curve compounds this issue.
However, there is exhaustive documentation with many tutorials (as texts and videos), best practices, and user guides on building various models across different domains.
Also, if an experiment is implemented in PyTorch Lightning, there are a lot of helpful tweaks and techniques to improve or speed up the training.
So that it can be worthwhile even when using pre-built models.
These facilitation features include tweaks like training with half-precision, automatic tuning of the learning rate, and integrations into hyperparameter tuning frameworks or creating command-line interfaces to control the parameters.
In addition to that, there is support for different computational backends that help to dispatch the training on multiple accelerators like GPUs and TPUs.
If these features are not enough, there is a growing ecosystem of third-party extensions, widening the scope and functionality of the framework



## Classes
From a technical point of view, Pytorch Lightning provides an API composed of three different main classes, dividing the training process into a sequence of single atomic steps.
These classes implement the model, the logic for storing and processing the training data, and the training process itself.

### `LightningModule`

A subclass of a `LightningModule` implements the model.
A `LightningModule` is an extended version of PyTorch's `nn.Module` class.
`nn.Modules` are the basic building blocks of neural networks in PyTorch. In essence, they store a set of parameters, for example, weights of a single layer alongside with `.forward`-method that defines the computational logic when data flows through the module. They are designed to work recursively. One module can be composed of several submodules so that each building block of a neural network, starting from single layers up to a network, can be implemented in this one class.
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

network = nn.Sequential(
    DenseLayer(512, 16),
    nn.ReLU(),
    DenseLayer(16, 8),
    nn.ReLU(),
    DenseLayer(8, 2)
)

inputs = torch.randn(8, 512)  # Batchsize 8
outputs = network(inputs)
print(outputs.size())
print(issubclass(nn.Sequential, nn.Module))
```

By chaining multiple instances of the dense layer in a `nn. Sequential` class, it is possible to create a simple feed-forward network. This network is again a subclass of the `nn.Module` class.

A `LightningModule` is intended to replace the outmost `nn.Module` instance of a model, meaning the one that holds a complete network.
It extends the `nn.Module` class with new methods, designed to structure not only the logic of a single forward pass but also other steps like a complete train-, test- or validation-steps.
With this extension, it becomes possible to define a single forward step through the network and how the models should be trained and tested as well.
In essence, it provides a way to incorporate the training loop into the model itself.
This strategy has one massive advantage over the standard PyTorch practice of writing an external function that implements the training loop.
It helps to make the model self-contained, meaning that it holds all necessary logic itself. This property alleviates sharing models since only one class carries all information to train and test the model.

#### Training, Testing, Validation

The methods for training, testing and prediction are called `.train_step`,  `.validation_step` and `.test_step` respectively.
They all define how a single batch of data should be handled for these steps.
Typically the `.train_step`-method computes the loss score, and the other two methods compute other validation metrics.
Design-wise, only the `.train_step`-method is required to return the loss averaged loss score for the current batch.
The test- and validation methods are not required to return anything. Instead, they can use the built-in logging capabilities of the `LightningModule`.
Similar to the `.train_step`-method, they should return their scores averaged over the complete batch too.

#### Model Hyperparameters and checkpoints

Much effort has been put into organizing the hyperparameters of an experiment.
Like the train and test routines, PyTorch Lightning binds all hyperparameters that control the model directly to the object.
This strategy ensures that saved models also contain the combination of parameters used for training.
There are two ways to define the hyperparameters of a model.
By default, each argument in the signature of the model's constructor is regarded as a hyperparameter.
By calling a `.save_hyperparameters`-method in the constructor, these arguments get serialized into a `.hparams`-attribute.
The `.hparams`-attribute is saved as a `YAML`-file for each checkpoint of the model, making it easy to see which parameters were used without loading the whole model.
If the constructor contains non-hyperparameters arguments, these can be excluded from serialization using the saving method's `ignore` flag.

Another more explicit way of defining the parameters of a model is to store them all in a dictionary into the constructor and to pass this dictionary to the `-save_hyperparameters`-method.
This strategy is suitable in cases where many arguments of the constructor are non-hyperparameters.

#### Logging

Logging in PyTorch-Lightning is a two-stage procedure.
Inside the `LightningModule`, various metrics can be logged at different steps while training using the `.log`- or `log_dict`-methods.
The `.log`-methods can log a single score, while the `log-dict`-method can log multiple scores stored in a dictionary (with names as keys and the score as values).
These logs are extracted by `Trainer` and written out in various formats (see Trainer section for further details.)
One benefit of using an autonomous logging function is that it gives flexibility to the user in deciding when to log which metrics, for example, making it possible to log something only when a condition applies.


### `LighningDataModule`

`PyTorchLightning` also comes with a custom solution to bundle data-related operations into a single object.
It is called `LighningDataModule` and should contain the code to load and prepare the data for training and testing.
A class derived from `LighntningDataModule` must implement four required methods.
The `.prepare_data`-method should implement all steps required to load the data and convert it into a correct representation for the model.
To return the splits for training, testing and evaluation, there are `.train|.test|.val_loader`-methods. Each of them has to return a `DataLoader` object.
Like the `LightningModule`, its data counterpart has the advantage of holding all code to load and prepare the data, which alleviates distribution and publication.
Another key feature of the `LightningDataModule` is its ability to adapt to distributed environments.
While the `.prepare_data`-method is called once at the beginning of the training, there are also additional `.setup`- and `teardown`-methods.
These methods can define operations pre- or post-training data-preparation steps that must be performed independently on each accelerator.

### `Trainer`

The `Trainer` object handles the actual training.
It receives the model and data (wrapped in Lightning modules) alongside all training-specific hyperparameters, like the number of epochs, the devices to train on, or a list of loggers to log the progress.

The `Trainer` exposes four high-level methods to the user. Each of them triggers either the training, the validation, the prediction of unseen instances, and the hyperparameter-tuning.
Like the `LightningModule`, an instance of the `Trainer` is initialized with all hyperparameters relevant to the training, like the batch size or a number of epochs.

The different stages of the training (training and testing)

### Extending the `Trainer`

In contrast to the `LightningModule` and `LightningDataModule` the `Trainer` itself is not intended to be customized in any way. Since their respective objects contain all model or data-related code, the `Trainer` is better kept untouched.
Instead, if necessary, the functions of the `Trainer` can be extended with callbacks and plugins. Both of them can add custom operations to different stages of the training.
Callbacks implement steps that are not strictly necessary for training. Instead, they can be used to define things like logging or applying non-essential operations to the model (i.e., weigh pruning after each epoch) that add new functions but are not required to perform training.
On the other hand, plugins are meant to extend the `Trainer` with new functionalities like adding support for new accelerators or computational backends. So by their scope, they are meant to be used by experienced users who need to extend the Trainer.
However, since their API is still in beta and subject to changes in the future, it should be used with caution.

Also, it contains a handful of tweaks to improve the results, like gradient accumulation or gradient clipping
In addition to that, the `Trainer` supports tuning the learning rate and batch size out of the box. Both features have to be enabled while initializing the `Trainer` and can be invoked by calling the `.tune`-method.
### Logging

While the model defines what measures are logged, the `Trainer` is responsible for writing out these logs.
By default, it logs the standard output.
In addition to that, it can be extended with additional loggers.
PyTorch Lightning provides built-in loggers that log the progress to Tensorboard or other services like Weights and Biases.
Further loggers, can be implemented using the Logger base class.
Additional loggers are passed to the `Trainer` during initialization.

## CLI Interface

PyTorch Lightning supports the creation of command-line interfaces through the `LightninArgumentParser` class.
This class is an extended version of the `jsonargparse` and can parse the arguments of Lightning classes and other classes out-of-the-box.
This feature enables adding parameters of different modules to the parser effortlessly.

If more flexibility is needed, for example, when only some parameters of an object should be added to the parser, the best practice is to add a method to the object, which adds these arguments to the parser.

```python
:name: lightning-parser-listing
parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
parser.add_class_arguments(TensorBoardLogger, nested_key="tensorboard")
parser.add_lightning_class_args(Trainer, "trainer")
parser = PlLanguageModelForSequenceOrdering.add_model_specific_args(parser)
```

## Implementation

## Model

## Transformers

Since we do not build our own model, we need to load the pretrained transformer in the constructor of the model.
To be able to load different models, we introduce the name of the model as hyperparameters.
Since the model is pretrained, we only have to specify two other hyperparameters, namely the learning rate and the id of the target token.
Because Huggingface models are also subclasses of the  `nn.Module` class, loading the transformer model works flawlessly, and the language model is recognized as a submodule of the `PlLanguageModelForSequenceOrdering` class.

```python bla
class PlLanguageModelForSequenceOrdering(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.base_model = AutoModelForTokenClassification.from_pretrained(
            self.hparams["model_name_or_path"],
            return_dict=True,
            output_hidden_states=True,
            num_labels=1,
        )
```

Next, we define a single forward step. This is fairly simple since the only thing we need to do is exclude the labels from the inputs for the language model and pass the rest of the input data to the language model to obtain the ouputs.

```python
    def forward(self, inputs: Dict[Any, Any]) -> Dict[Any, Any]:
        # We do not want to compute token classification loss, so we remove the labels temporarily
        labels = inputs.pop("labels")
        outputs = self.base_model(**inputs)

        # And reattach them later on ...
        inputs["labels"] = labels
        return outputs
```

Since we want to compute the loss while training and while validating the model, we factor out the loss function into a separate method.
Implementation-wise, the loss function is only slightly variated from the original implementation. The only changes are that we retrieve the target token id from the hyperparameters of the model.
Also, we draw inspiration from the `transformers` API and add a custom version of the forward method. This method computes both the forward step and the loss. The loss is then attached to the output of the model.

``` python
    def _compute_loss(self, batch_labels, batch_logits, batch_input_ids) -> float:
        # Since we have varying number of labels per instance, 
        # we need to compute the loss manually for each one.
        loss_fn = nn.MSELoss(reduction="sum")
        batch_loss = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        for labels, logits, input_ids in zip(
            batch_labels, batch_logits, batch_input_ids
        ):

            # Firstly, we need to convert the sentence indices to regression targets.
            # To avoid exploding gradients, we norm them to be in range 0 <-> 1.
            # labels = labels / labels.max()
            # Also we need to remove the padding entries (-100).
            true_labels = labels[labels != -100].reshape(-1)
            targets = true_labels.float()

            # Secondly, we need to get the logits 
            # from each target token in the input sequence
            target_logits = logits[
                input_ids == self.hparams["target_token_id"]
            ].reshape(-1)

            # Sometimes we will have less target_logits 
            # than targets due to truncation of the input.
            # In this case, we just consider as many targets as we have logit.
            if target_logits.size(0) < targets.size(0):
                targets = targets[: target_logits.size(0)]

            # Finally we compute the loss for the current instance 
            # and add it to the batch loss.
            batch_loss = batch_loss + loss_fn(targets, target_logits)

        # The final loss is obtained by averaging 
        # over the number of instances per batch.
        loss = batch_loss / batch_logits.size(0)

        return loss

    def _forward_with_loss(self, inputs):
        outputs = self(inputs)

        # Get sentence indices
        batch_labels = inputs["labels"]
        # Get logits from model
        batch_logits = outputs["logits"]
        # Get logits for all cls tokens
        batch_input_ids = inputs["input_ids"]

        loss = self._compute_loss(
            batch_labels=batch_labels,
            batch_logits=batch_logits,
            batch_input_ids=batch_input_ids,
        )
        outputs["loss"] = loss

        return outputs
```

Using the `_foward_with_loss`-method implementing the `training_step`-method becomes relatively simple.
The only thing left to do inside this method is to log the training loss in order to be able to monitor the progress during training.

```python
    def training_step(self, inputs: Dict[Any, Any], batch_idx: int) -> float:
        outputs = self._forward_with_loss(inputs)
        loss = outputs["loss"]
        self.log("loss", loss, logger=True)
        return loss
```

Like the `_compute_loss`-method, we only need to slightly adapt the validation metrics' computation to use the model's hyperparameters.
Since we want to compute the identical scores for testing and validation, we can also use the `validation_step`-method for testing.

```python
    def validation_step(self, inputs, batch_idx):
        outputs = self._forward_with_loss(inputs)

        # Detach all torch.tensors and convert them to np.arrays.
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = value.detach().cpu().numpy()
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.detach().cpu().numpy()

        # Get sentence indices
        batch_labels = inputs["labels"]
        # Get logits from model
        batch_logits = outputs["logits"]
        # Get logits for all cls tokens
        batch_input_ids = inputs["input_ids"]

        metrics = defaultdict(list)
        for sent_idx, input_ids, logits in zip(
            batch_labels, batch_input_ids, batch_logits
        ):
            sent_idx = sent_idx.reshape(-1)
            input_ids = input_ids.reshape(-1)
            logits = logits.reshape(-1)

            sent_idx = sent_idx[sent_idx != 100]
            target_logits = logits[input_ids == self.hparams["target_token_id"]]
            if sent_idx.shape[0] > target_logits.shape[0]:
                sent_idx = sent_idx[: target_logits.shape[0]]

            # Calling argsort twice on the logits 
            # gives us their ranking in ascending order.
            predicted_idx = np.argsort(np.argsort(target_logits))
            tau, pvalue = kendalltau(sent_idx, predicted_idx)
            acc = accuracy_score(sent_idx, predicted_idx)
            metrics["kendalls_tau"].append(tau)
            metrics["acc"].append(acc)
            metrics["mean_logits"].append(logits.mean().item())
            metrics["std_logits"].append(logits.std().item())

        metrics["loss"] = outputs["loss"].item()

        # Add val prefix to each metric name and compute mean over the batch.
        metrics = {
            f"val_{metric}": np.mean(scores).item()
            for metric, scores in metrics.items()
        }
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True, on_step=True)
        return metrics

    def test_step(self, inputs, batch_idx):
        return self.validation_step(inputs, batch_idx)
```

Lastly, we need to implement the `configure_optimizers`-method and add the model's hyperparameter to the parser via the `add_model_specific_args`-method.

```python
    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams["lr"])

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group(
            "PlLanguageModelForSequenceOrdering"
            )
        parser.add_argument(
            "--model.model_name_or_path", type=str, default="bert-base-cased"
        )
        parser.add_argument("--model.lr", type=float, default=3e-5)
        parser.add_argument("--model.target_token_id", type=int, default=101)
        return parent_parser
```

## Data

In contrast to the model class, we design our version of the `LightningDataModule` to work with any Huggingface `Dataset`.
Most of the work is done by the `.prepare_data`-method, which implements the processing pipeline for the contained dataset
Firstly, it applies all functions to prepare the data via the `.map`-method of the `Dataset` class.
Afterward, the text data will be tokenized using the passed instance of the tokenizer.
Lastly, it is ensured that the dataset's column containing the target is named `labels` to be compliant with standard `transformers` models.
Additionally, we implement a method to use the map functionalities of the contained dataset directly.
This method allows the manipulation of the data manually since the `.prepare_data`-method is automatically executed by the `Trainer`.
The datasets wrapped in this class should already contain train-/ test- and validation-splits.
To create batches of the data, we use the default collation function of the transformers library but allow passing a custom collation function.

```python
class HuggingfaceDatasetWrapper(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        text_column: str,
        target_column: str,
        tokenizer: PreTrainedTokenizerBase,
        train_batch_size: int = 8,
        eval_batch_size: int = 16,
        mapping_funcs: List[Callable] = None,
        collate_fn: Callable = default_data_collator,
        train_split_name: str = "train",
        eval_split_name: str = "val",
        test_split_name: str = "test",
    ):
        super().__init__()
        self.dataset = dataset
        self.text_column = text_column
        self.target_column = target_column
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.mapping_funcs = mapping_funcs
        self.collate_fn = collate_fn
        self.train_split_name = train_split_name
        self.eval_split_name = eval_split_name
        self.test_split_name = test_split_name

    def prepare_data(self, tokenizer_kwargs: Dict[str, str] = None):
        # 1. Apply user defined preparation functions
        if self.mapping_funcs:
            for mapping_func in self.mapping_funcs:
                dataset = dataset.map(mapping_func, batched=True)

        # 2. Tokenize the text
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {
                "truncation": True,
                "padding": "max_length",
                "add_special_tokens": False,
            }
        self.dataset = self.dataset.map(
            lambda e: self.tokenizer(e[self.text_column], **tokenizer_kwargs),
            batched=True,
        )
        # 3. Set format of important columns to torch
        self.dataset.set_format(
            "torch", columns=["input_ids", "attention_mask", self.target_column]
        )
        # 4. If the target columns is not named 'labels' rename it
        try:
            self.dataset = self.dataset.rename_column(self.target_column, "labels")
        except ValueError:
            # target column should already have correct name
            pass

    def train_dataloader(self):
        return DataLoader(
            self.dataset[self.train_split_name],
            batch_size=self.train_batch_size,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset[self.eval_split_name],
            batch_size=self.eval_batch_size,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset[self.test_split_name],
            batch_size=self.eval_batch_size,
            collate_fn=self.collate_fn,
        )

    def map(self, *args, **kwargs):
        self.dataset = self.dataset.map(*args, **kwargs)

```

## Complete code

Once again, after factoring out the custom modules, the actual experiment can be implemented in relatively few lines of code.
To control the experiment via the command line, we use the `LightningArgumentParser`.
We initialize the parser with all arguments from the `Trainer,` `PlLanguageModelForSequenceOrdering`, and `HuggingfaceDatasetWrapper`.
Additionally, we add more parameters to give each run a name and control the batch sizes for training and testing.
Similar to implementing the experiment with the Huggingface `Trainer`, we need to make sure that the sentences contain the correct special tokens.
Replacing these tokens if necessary can be done using the `.map`-method of the `HuggingfaceDatasetWrapper`


```python
from os.path import basename
from datasets import load_from_disk
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningArgumentParser
from transformers import AutoTokenizer

from pl_modules import (
    HuggingfaceDatasetWrapper,
    PlLanguageModelForSequenceOrdering,
    so_data_collator,
)


def main(model_args, trainer_args, checkpoint_args, tensorboard_args, run_args):

    seed_everything(run_args["seed"])

    print("Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(model_args["model_name_or_path"])

    print("Loading datasets.")
    data = load_from_disk("../data/rocstories")

    # Downsampling for debugging...
    # data = data.filter(lambda _, index: index < 5000, with_indices=True)

    dataset = HuggingfaceDatasetWrapper(
        data,
        text_column="text",
        target_column="so_targets",
        tokenizer=tokenizer,
        mapping_funcs=[],
        collate_fn=so_data_collator,
        train_batch_size=run_args["train_batch_size"],
        eval_batch_size=run_args["val_batch_size"],
    )

    if tokenizer.cls_token != "[CLS]":
        print(
            f"Model does not a have a [CLS] token. Updating the data with token {tokenizer.cls_token} ..."
        )

        def replace_cls_token(entry):
            texts = entry["text"]
            replaced_texts = []
            for text in texts:
                replaced_texts.append(text.replace("[CLS]", tokenizer.cls_token))
            entry["text"] = replaced_texts
            return entry

        dataset = dataset.map(replace_cls_token, batched=True)
        model_args["target_token_id"] = tokenizer.cls_token_id

    print("Loading model.")
    model = PlLanguageModelForSequenceOrdering(hparams=model_args)

    print("Initializing trainer.")
    # Init logger
    tensorboard_logger = TensorBoardLogger(**tensorboard_args)

    # Init callbacks
    callbacks = []
    checkpoint_callback = ModelCheckpoint(**checkpoint_args)
    callbacks.append(checkpoint_callback)

    # Remove default args
    trainer_args.pop("logger")
    trainer_args.pop("callbacks")
    trainer = Trainer(logger=tensorboard_logger, callbacks=callbacks, **trainer_args)

    print("Start tuning.")
    trainer.tune(model=model, datamodule=dataset)

    print("Start training.")
    trainer.fit(model=model, datamodule=dataset)

    print("Start testing.")
    trainer.test()


if __name__ == "__main__":
    parser = LightningArgumentParser()
    group = parser.add_argument_group()
    group.add_argument("--run.run_name", type=str, default=basename(__file__))
    group.add_argument("--run.seed", type=int, default=0)
    group.add_argument("--run.train_batch_size", type=int, default=8)
    group.add_argument("--run.val_batch_size", type=int, default=16)

    parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
    parser.add_class_arguments(TensorBoardLogger, nested_key="tensorboard")
    parser.add_lightning_class_args(Trainer, "trainer")
    parser = PlLanguageModelForSequenceOrdering.add_model_specific_args(parser)

    args = parser.parse_args()

    model_args = args.get("model", {})
    trainer_args = args.get("trainer", {})
    checkpoint_args = args.get("checkpoint", {})
    tensorboard_args = args.get("tensorboard", {})
    run_args = args.get("run", {})

    main(model_args, trainer_args, checkpoint_args, tensorboard_args, run_args)

```

## Conclusion

Pytorch Lightning goal is not to hide complexity from the user. Instead, it provides an API that helps to structure the complexity into a sequence of single steps.
This approach is constructive when designing custom models from scratch or implementing new training regimes that differ from the standard training loop.
This flexibility comes at the cost of friendliness to beginners. People who have little experience with PyTorch itself will quickly be overwhelmed by PyTorch Lightning API with vast possibilities to customize steps manually.
Even though the documentation is extensive and covers nearly all aspects of the library in great detail, it can be frustrating sometimes that there are multiple ways to achieve the same behavior, and there is little to no guidance in choosing between the different parts.
Like most modern deep learning frameworks, PyTorch Lightning is rapidly evolving, and thus many parts of it are either in beta and subject to significant changes in the future or deprecated. Unfortunately, this is also noticeable when searching the web for further advice since many tips or tutorials quickly become outdated.
But despite these limitation for beginners, experienced user can really benefit from using PyTorch Lightning. Not only because of the additional features like built-in logging, tuning or other tweaks, but mainly because the well thought API enforces them to write self-contained models that contain all the logic for experimenting with them.
This enables sharing of model in an effortless way and also alleviates maintainability.
