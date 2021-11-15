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

```{code-cell} ipython3
:tags: ["remove-cell"]

from datasets import set_caching_enabled
set_caching_enabled(False)

import pprint
pp = pprint.PrettyPrinter(depth=6, compact=True)
print = pp.pprint
```

# Poutyne

Compared to the other two frameworks, Poutyne has a different scope.

Instead of trying to make the training of a fixed set of models as easy as possible like Huggingface `Trainer`, or facilitating the creation and training of custom models like PyTorch Lightning, it tries to bring the ease of the Keras API from the realms of Tensorflow to the world of PyTorch.
The benefits of the Keras API are its simplicity and orientation at well-established machine learning frameworks like Scikit-Learn.
This simplicity lowers the barrier of entry for beginners because it lowers the amount of time needed to get hands-on training for their first model.
The following exemplary listing shows the typical workflow in Poutyne.

```python
from poutyne import Model

...

network = make_network()
X_train, y_train = load_data(subset="train")
X_val, y_val = load_data(subset="validation")
X_test, y_test = load_data(subset="test")

model = Model(
    network,
    "sgd",
    "cross_entropy",
    batch_metrics=["accuracy"],
    epoch_metrics=["f1"],
    device="cuda:0"
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=64
)

results = model.evaluate(X_test, y_test, batch_size=128)
```

Like Keras, Poutyne automates many steps for standard cases like the configuration of the optimizer or the loss function.
However, Poutyne does not mimic the whole Keras API but only the training part.
The model’s creation still has to be done in plain PyTorch, which is generally a bit trickier than Keras because the dimensions of all layers have to be chosen manually.
In addition to the training functions, Poutyne also provides utilities to conduct and save whole experiments and utilities for creating checkpoints, logging, scheduling of the learning rate, and multi-device training.

## Classes

### Model

The `Model` class is intended to handle the training of any neural network
Technically, it wraps a neural network alongside an optimizer, loss function, and validation metrics.
To train the model, it exposes different variants of the `fit`-method, each of which can process the training data in another format.
The standard `.fit`-methods expects the data as a list of batches, while the `fit_dataset` method can directly work on PyTorch `Datasets`.
The `fit_generator` can operate on generators that yield the data batch by batch as a third option.
In addition to that, the `fit`-methods also receive other hyperparameters to control the training.

The `evaluate`-method computes the loss and all other metrics on unseen data without doing backpropagation.
If only the predictions of the network are needed, the `predict`-method can be used.
Similar to the variations of the `fit`-methods, these methods are offered in different versions too.

### Experiment

The `Experiment` class is an extended version of the `Model` class that comes with helpful additions for conducting deep learning experiments.
Like the `Model` class, an `Experiment` is equipped with the neural network, optimizer, loss function, and metrics into a single object and has methods to start the training, evaluation, or prediction.
In contrast to the `Model` class, which is solely designed to train a model, the `Experiment` class provides additional features to organize and track the training
For example, it supports logging the progress to various formats, like a CSV table or Tensorboard.
Monitoring allows the `Experiment` class to save checkpoints of the model that perform best with respect to one of the validation metrics.
Also, it saves all the intermediate results and tracked values to the disk.

For the two primary task types, classification and regression, the experiment automatically configures all metrics and the loss function if these tasks are specified in the `task` parameter when initializing the `Experiment`.

### Data

Poutyne is data agnostic meaning, that it does not provide any tooling to load, process, and store the training data.
The only requirements are that the data comes in one of the supported formats and that each batch consists of two objects: one that holds the training data and one that contains the label.
To compare the model’s output with the labels, it has to be in the same format.

## Additional Features

#### Metrics

Poutyne has a custom API for implementing metrics.
It distinguishes between two types of metrics, batch metric and epoch metrics.
Batch metrics are computed per batch and averaged to obtain the results for one single epoch.
Epoch metrics are computed on the gathered results of one entire epoch. Thus, they are a good choice for measures that would suffer from averaging over the batch results, like the F-score.
Poutyne provides predefined metrics for both types. But, unfortunately, they only cover classification tasks.
There are two options to add other metrics. Either they have to be implemented manually or taken from Scikit-Learn and made compatible using a built-in wrapper class.
Metrics are passed to `Model` or `Experiment` while their initialization.

#### Callbacks

Callbacks are intended to extend the functions of the `Model` or `Experiment` class. Like the callbacks from the other frameworks, they have access to the model’s current state and can perform actions at various steps while training.
There are many predefined callbacks available that perform all kinds of tasks, ranging from logging, keeping track of gradients, scheduling the learning, creating checkpoints, to sending notifications to inform clients about the progress of the training.

## Implementation

Even though the `transformers` is not compatible with vanilla Poutyne, integrating it does not require complicated changes.
Most of the required adaptions change the data in order to convert betweeN the dictionary-based data model of the `transformers` library and Poutyne’s more classical `X, y` format for input data and targets.
Since these changes are task agnostic, we factored most of these adaption tools out of the project into a small standalone library. [^gh-link]

[^gh-link]: [poutyne-transformers](https://github.com/LennartKeller/poutyne-transformers)

### Data

To convert the data for an experiment from the Hugginface `Dataset` format into a Poutyne compliant representation, we create a custom data collator.
The main task is of the collator is to convert each batch of dictionaries into batches of tuples of training data and targets.
To so the `TransformersCollator` can is initialized with one or multiple names of the target `y_keys` in the input dictionaries. These fields then get copied into the target object. This object is either a tensor, if only a single key was specified or into a dictionary of keys. Additionally, the `remove_labels` parameter can be used to either remove the labels after copying them into the target obejct. By default, they are retained in the input data. This enables to use the internal computation of the loss of standard models, while also being able to use the built-in metrics of Poutyne for monitoring the training.
Other collation operations are handled by the default collator from `transformers` or by a custom function.

```python
from typing import Any, Callable, Dict, List, Tuple, Union

import torch
from transformers import default_data_collator


class TransformerCollator:
    def __init__(
        self,
        y_keys: Union[str, List[str]] = None,
        custom_collator: Callable = None,
        remove_labels: bool = False,
    ):
        self.y_keys = y_keys
        self.custom_collator = (
            custom_collator if custom_collator is not None else default_data_collator
        )
        self.remove_labels = remove_labels

    def __call__(self, inputs: Tuple[Dict]) -> Tuple[Dict, Any]:
        batch_size = len(inputs)
        batch = self.custom_collator(inputs)
        if self.y_keys is None:
            y = torch.tensor(float("nan")).repeat(batch_size)
        elif isinstance(self.y_keys, list):
            y = {
                key: batch.pop(key)
                if "labels" in key and self.remove_labels
                else batch.get(key)
                for key in self.y_keys
            }
        else:
            y = batch.pop(self.y_keys) if self.remove_labels else batch.get(self.y_keys)
        return batch, y
```

### Model

As stated in the Prerequisites chapter, a model’s tokenizer returns a dictionary of data that contains all data required to be fed into the language model unpacked as keyword arguments.

```{code-cell} ipython3
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("bert-base-cased")

inputs = tokenizer("Poutyne is inspired by Keras", return_tensors="pt")
print(model(**inputs).keys())
```

Poutyne instead passes the data to the model in the same format it receives it.
To make sure that the data is unpacked and passed to the model correctly, we create wrapper class.

```python
from typing import Any, Dict
from torch import nn
from transformers import PreTrainedModel


class ModelWrapper(nn.Module):
    def __init__(self, transformer: PreTrainedModel):
        super().__init__()
        self.transformer = transformer

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.transformer)})"

    def forward(self, inputs) -> Dict[str, Any]:
        return self.transformer(**inputs)

    def save_pretrained(self, *args, **kwargs) -> None:
        self.transformer.save_pretrained(*args, **kwargs)
```

It is also a subclass of the `nn. Module` to ensure that all parameters of the contained model can be accessed.
Apart from the data handling, this class does not more than exposing the custom `save_pretrained`-model of the underlying `transformers` model.
This way, it is possible to create checkpoints of the trained model that can be loaded and used in the `transformers` ecosystem.

### Loss

In Poutyne the loss function receives the output of the model and the targets.
When using default models, neither of both has to be used to obtain the loss, since we can extract the internal loss from the model's output.
In our case we have to implement a function that computes the loss on our own.
Since we do not have access to the model or the tokenizer, we have to create a loss function that stores the id of the current target token. For that, we opt for creating a class that holds this id as an attribute and computes the loss via its `__call__` method.

```python
class PoutyneSequenceOrderingLoss:
    def __init__(self, target_token_id):
        self.target_token_id = target_token_id

    def __call__(self, outputs, targets) -> float:
        batch_labels = targets["labels"]
        batch_logits = outputs["logits"]
        batch_input_ids = targets["input_ids"]

        # Since we have varying number of labels per instance, we need to compute the loss manually for each one.
        loss_fn = nn.MSELoss(reduction="sum")
        batch_loss = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        for labels, logits, input_ids in zip(
            batch_labels, batch_logits, batch_input_ids
        ):
            # Firstly, we need to convert the sentence indices to regression targets.
            # To avoid exploding gradients, we norm them to be in range 0 <-> 1
            # Also we need to remove the padding entries (-100)
            true_labels = labels[labels != -100].reshape(-1)
            targets = true_labels.float()

            # Secondly, we need to get the logits from each target token in the input sequence
            target_logits = logits[input_ids == self.target_token_id].reshape(-1)

            # Sometimes we will have less target_logits than targets due to trunction of the input
            # In this case, we just consider as many targets as we have logits
            if target_logits.size(0) < targets.size(0):
                targets = targets[: target_logits.size(0)]

            # Finally we compute the loss for the current instance and add it to the batch loss
            batch_loss = batch_loss + loss_fn(targets, target_logits)

        # The final loss is obtained by averaging over the number of instances per batch
        loss = batch_loss / batch_logits.size(0)

        return loss
```

### Metrics

Unlike the Huggingface `Trainer`, which expects all external metrics a single function to compute them all at once, in Poutyne the `Model` or `Experiment`classes are equipped with a list of functions, each of which represents a different metric.
Like the loss, functions that compute other performance metrics receive the model's output alongside the targets (extracted by collation function).
Because `transformer` models return not only the logits or predictions of a model but also other things, it is not possible to use Poutynes built-in metrics out of the box.
They expect the output to be a single tensor containing the logits of the model, so we create a wrapper for metric functions that extracts them from the output and passes them to the metric.

```{code-cell} ipython3
from typing import Any, Callable, Dict

class MetricWrapper:
    def __init__(self, metric: Callable, pred_key: str = "logits", y_key: str = None):
        self.metric = metric
        self.pred_key = pred_key
        self.y_key = y_key
        self._set_metric_name(metric)

    def _set_metric_name(self, metric):
        self.__name__ = metric.__name__

    def __call__(self, outputs: Dict[str, Any], y_true: Any):
        y_pred = outputs[self.pred_key]
        if self.y_key is not None:
            y_true = outputs[self.y_key]
        return self.metric(y_pred, y_true)
```

Since the logging components of Poutyne infer the name of the metric by assessing the class name of their functions, we need to set the `__name__`-attribute of our wrapper instance with the name of the contained metric.

To implement our sentence ordering metrics, we implement a function that returns a function for each of them.
These function can then be wrapped by t

```{code-cell} ipython3
import numpy as np
from collections import defaultdict
from functools import partial
from sklearn.metrics import accuracy_score
from scipy.stats import kendalltau

def make_compute_metrics_functions(target_token_id) -> Callable:
    def compute_ranking_func(
        outputs: Dict, targets: Any, metric_key: str
    ) -> Dict[str, float]:
        batch_sent_idx = targets["labels"].detach().cpu().numpy()
        batch_input_ids = targets["input_ids"].detach().cpu().numpy()
        batch_logits = outputs.detach().cpu().numpy()

        metrics = defaultdict(list)
        for sent_idx, input_ids, logits in zip(
            batch_sent_idx, batch_input_ids, batch_logits
        ):
            sent_idx = sent_idx.reshape(-1)
            input_ids = input_ids.reshape(-1)
            logits = logits.reshape(-1)

            sent_idx = sent_idx[sent_idx != 100]
            target_logits = logits[input_ids == target_token_id]
            if sent_idx.shape[0] > target_logits.shape[0]:
                sent_idx = sent_idx[: target_logits.shape[0]]
            # Calling argsort twice on the logits gives us their ranking in ascending order
            predicted_idx = np.argsort(np.argsort(target_logits))
            tau, pvalue = kendalltau(sent_idx, predicted_idx)
            acc = accuracy_score(sent_idx, predicted_idx)
            metrics["kendalls_tau"].append(tau)
            metrics["acc"].append(acc)
            metrics["mean_logits"].append(logits.mean())
            metrics["std_logits"].append(logits.std())
        metrics = {metric: np.mean(scores) for metric, scores in metrics.items()}
        return metrics[metric_key]

    metrics = []
    for metric in ("acc", "kendalls_tau", "mean_logits", "std_logits"):
        metric_func = partial(compute_ranking_func, metric_key=metric)
        metric_func.__name__ = metric
        metrics.append(metric_func)
    return metrics

metrics = [
        MetricWrapper(func)
        for func in make_compute_metrics_functions(0)
    ]
print([metric.__name__ for metric in metrics])
```

Additionally, we add two functions to track the mean and standard deviation of the logits to monitor whether the regression can fit the desired indices or only learns their average, which lies around `2.5`.
### Complete code

Once again, we factor out our adaptions into an external module and implement the rest of the experiment.
Due to Poutyne's lack of tooling for creating a command-line interface, this experiment is only configurable via hard-coding the parameters into the source.
The rest of the code is mainly similar to the other two frameworks.

```python
from poutyne.framework import experiment
from torch.optim import AdamW
from poutyne import (
    Model,
    set_seeds,
    TensorBoardLogger,
    TensorBoardGradientTracker,
    Experiment,
)
from poutyne_transformers import ModelWrapper, MetricWrapper, TransformerCollator
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_from_disk
from poutyne_modules import (
    make_tokenization_func,
    PoutyneSequenceOrderingLoss,
    make_compute_metrics_functions,
    so_data_collator,
)


if __name__ == "__main__":
    set_seeds(42)

    MODEL_NAME_OR_PATH = "bert-base-cased"
    LEARNING_RATE = 3e-5
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 16
    DEVICE = "cuda:1"
    N_EPOCHS = 3
    SAVE_DIR = "experiments/rocstories/bert"

    print("Loading model & tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    transformer = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME_OR_PATH, return_dict=True, num_labels=1
    )

    print("Loading & preparing data.")
    dataset = load_from_disk("../data/rocstories/")

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

    tokenization_func = make_tokenization_func(
        tokenizer=tokenizer,
        text_column="text",
        add_special_tokens=False,
        padding="max_length",
        truncation=True,
    )
    dataset = dataset.map(tokenization_func, batched=True)

    dataset = dataset.rename_column("so_targets", "labels")

    dataset = dataset.remove_columns(
        ["text", "storyid", "storytitle"] + [f"sentence{i}" for i in range(1, 6)]
    )
    dataset.set_format("torch")

    collate_fn = TransformerCollator(
        y_keys=["labels", "input_ids"],
        custom_collator=so_data_collator,
        remove_labels=True,
    )

    train_dataloader = DataLoader(
        dataset["train"], batch_size=TRAIN_BATCH_SIZE, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        dataset["val"], batch_size=VAL_BATCH_SIZE, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        dataset["test"], batch_size=VAL_BATCH_SIZE, collate_fn=collate_fn
    )

    print("Preparing training.")
    wrapped_transformer = ModelWrapper(transformer)
    optimizer = AdamW(wrapped_transformer.parameters(), lr=LEARNING_RATE)
    loss_fn = PoutyneSequenceOrderingLoss(target_token_id=tokenizer.cls_token_id)

    metrics = [
        MetricWrapper(func)
        for func in make_compute_metrics_functions(tokenizer.cls_token_id)
    ]

    writer = SummaryWriter("runs")
    tensorboard_logger = TensorBoardLogger(writer)
    gradient_logger = TensorBoardGradientTracker(writer)

    experiment = Experiment(
        directory=SAVE_DIR,
        network=wrapped_transformer,
        device=DEVICE,
        logging=True,
        optimizer=optimizer,
        loss_function=loss_fn,
        batch_metrics=metrics,
        monitoring=True,
        monitor_metric="val_loss",
        monitor_mode="min",
    )

    experiment.train(
        train_generator=train_dataloader,
        valid_generator=val_dataloader,
        epochs=N_EPOCHS,
        save_every_epoch=True,
    )
```

## Conclusion

Poutyne provides a well thought and, most of all easy to understand framework to train neural networks.
Like its conceptual role model Keras, this simplicity is achieved by strict design decisions, like the `X, y` format for data.
While this strictness is helpful for beginners because they only have to learn one way of doing things, it comes at the cost of being hard to adapt to other frameworks or unintended tasks.
Luckily, the necessary steps to adapt it to `transformers` and our task are simple and can be reused for most other cases.
Since Poutynes mimics the Keras-API, its additional features are much more limited than the other frameworks.
Depending on the use-case this limited scope might be a deal-breaker for experienced users or complex tasks, but on the other hand, it makes getting started with the framework much more manageable.
This accessibility is underlined by the documentation's quality, which covers all aspects of the framework in concise and easily understandable manners without losing itself in the depths of technical details.
Yet, there is also potential for further improvements.
The lack of any support for creating-command line interfaces could force users to migrate to another framework as soon as they need to retrain a model regularly.
Currently, the scope of the framework is heavily skewed towards sequence classification tasks. For example, all built-in metrics measure the quality of a classification model.
Widening the range of tasks that could be implemented without further extensions would help beginners get into deep learning.
A possible improvement that falls more into the category of wishful thinking would be that Poutyne would mimic not only the training parts of the Keras API. If Poutyne would also introduce the ease of building neural networks without manually adjusting each layer's dimensionality manually, this would be a major contribution to the whole PyTorch community.
