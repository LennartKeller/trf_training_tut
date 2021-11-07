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

In contrast to the Huggingface `Trainer`, which handles the complete training itself, PyTorch Lightning takes a different approach.
Pytorch Lightning aims not only at handling the training but also at structuring the creation of a model itself.
Its main goal is not to hide complexity from the user but to provide a well structured model of building a model and implementing all other steps of a deep learning experiment in a consistent manner.
In its philosophy, a model and its inference-, training and prediction logic are not separate aspects that can be exchanged independently.
Instead, it binds them directly to the model itself.
In doing so, PyTorch-Lightning does not make many assumptions on the nature of the model or the training itself and thus allows maximum flexibility.
However, this flexibility comes at the cost that the user again is required to implement a lot of things manually.
This approach has a quite steep learning curve, especially for beginners, but PyTorch Lightning provides exhaustive documentation with many tutorials (as texts and videos), best practices, and user guides on building various models across different domains.

From a technical point of view, Pytorch Lightning provides an API composed of three different main classes, dividing the training process in a sequence of single atomic steps. 
Once a model follows this API, PyTorch Lightning it offers a lot of valuable tweaks like training with half-precision, automatic tuning of the learning rate, and integrations into hyperparameter tuning frameworks or creating command-line interfaces to be used out of the box.
Also, it supports different backends to train not only on GPUs but also on other accelerators like TPUs.
Some of these backends even support distributed training across different machines or processing nodes in a cluster.

Also there is a growing ecosystem of third party extensions that easily be used to extend the scope of the framework.

## Classes

A PyTorchLightning based projects comprises three main classes that implement the model, the storing and processing of the training data and the training process itself.
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

The `.forward`-method only defines the flow of the data through the model. For the logic of training, validation and testing there are three methods called `.train_step`,  `.validation_step` and `.test_step`.
They all define how a single batch of data should be handled at these steps.
Typically the `.train_step`-method computes the loss score and the other two methods compute other validaiton metrics.

### Optimizer

<!--In contrast to plain `PyTorch,` the optimizer is not regarded as an external object. Instead, it is directly bound to a model and its configuration is moved into the `configure_optimizer`-method of the model.-->

In PyTorch Lightnings view of a model the optimizer is not an external object but also a part of the model. Its configuration should be handled by the `.configure_optimizer`-method. This method must return an instance of any optimizer from PyTorch, that was initialized with the models parameters. This way the learning rate becomes part of the models hyperparameters.

### Model Hyperparameters

In `PyTorchLightning` there a lot of effort has been done to organize the hyperparameters of an experiment.
Most notably, there is a strict seperation of hyperparameters that diretly belong the the models itself or other than ones that control different aspect the training.
Model specific parameters are bound to the instance of the `Lightning 

For example, Hyperparameters that affect the model (like the number of hidden layers) are directly assigned to the model and saved with each checkpoint.
By default, each argument of the constructor of a `LighntningModule` is considered to be a hyperparameter.
By calling the `.save_hyperparameters`-method in the constructor, these arguments are serialized into a `.hparams`-attribute.
This strategy ensures that while loading an old checkpoint, it is entirely transparent which hyperparameters were used to train it.

```{code-cell} ipython3
# MORE TO COME HERE
```

### Logging

Since the logic of training and validation is bound to a model, logging must also be implemented here.
A `LighntingModule` has to possible options for logging metrics. A `.log`-methods that can log a single score or a `log-dict`-methods that can be used to log multiple scores stored in a dictionary (with names as keys and the score as values).
These methods can be used from any method making them not only feasible for logging validation but also training scores or other things.

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


## Implementation

## Model

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

    def forward(self, inputs: Dict[Any, Any]) -> Dict[Any, Any]:
        # We do not want to compute token classificaiton loss so we remove the labels temporarily

        labels = inputs.pop("labels")
        outputs = self.base_model(**inputs)

        # # Compute logits for each token in the input squence
        # last_hidden_state = outputs['last_hidden_state']
        # logits = self.linear(last_hidden_state)
        # outputs['logits'] = logits

        # And reattach them later on ...
        inputs["labels"] = labels
        return outputs

    def _compute_loss(self, batch_labels, batch_logits, batch_input_ids) -> float:
        # Since we have varying number of labels per instance, we need to compute the loss manually for each one.
        loss_fn = nn.MSELoss(reduction="sum")
        batch_loss = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        for labels, logits, input_ids in zip(
            batch_labels, batch_logits, batch_input_ids
        ):

            # Firstly, we need to convert the sentence indices to regression targets.
            # To avoid exploding gradients, we norm them to be in range 0 <-> 1
            # labels = labels / labels.max()
            # Also we need to remove the padding entries (-100)
            true_labels = labels[labels != -100].reshape(-1)
            targets = true_labels.float()

            # Secondly, we need to get the logits from each target token in the input sequence
            target_logits = logits[
                input_ids == self.hparams["target_token_id"]
            ].reshape(-1)

            # Sometimes we will have less target_logits than targets due to trunction of the input
            # In this case, we just consider as many targets as we have logits
            if target_logits.size(0) < targets.size(0):
                targets = targets[: target_logits.size(0)]

            # Finally we compute the loss for the current instance and add it to the batch loss
            batch_loss = batch_loss + loss_fn(targets, target_logits)

        # The final loss is obtained by averaging over the number of instances per batch
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

    def training_step(self, inputs: Dict[Any, Any], batch_idx: int) -> float:
        outputs = self._forward_with_loss(inputs)
        loss = outputs["loss"]
        self.log("loss", loss, logger=True)
        return loss

    def validation_step(self, inputs, batch_idx):
        outputs = self._forward_with_loss(inputs)

        # Detach all torch.tensors and convert them to np.arrays
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

            # Calling argsort twice on the logits gives us their ranking in ascending order
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
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)
        return metrics

    def test_step(self, inputs, batch_idx):
        return self.validation_step(inputs, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams["lr"])

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("PlLanguageModelForSequenceOrdering")
        parser.add_argument(
            "--model.model_name_or_path", type=str, default="bert-base-cased"
        )
        parser.add_argument("--model.lr", type=float, default=3e-5)
        parser.add_argument("--model.target_token_id", type=int, default=101)
        return parent_parser
```

## Data

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
        # 4. If the target columns is not named 'labels' change that
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
