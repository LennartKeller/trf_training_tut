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

# Huggingface Trainer

Since Huggingface proclaimed goal is to provide an environment to develop and train all sorts of language models, they also ship a solution for training models.
It is called the `Trainer` and comes with the `transformers` library itself.
Of course, it is profoundly integrated into the Huggingface-ecosystem and can train most `transformers` models out of the box.

## Classes

### Trainer

Design-wise, the `Trainer` is one single class that handles the training end-to-end.
Its configuration is outsourced to a `TrainingArguments` class that stores all relevant parameters for training.
These arguments are passed to the `Trainer`  alongside a model and a dataset during initialization.
The training then starts automatically.
Since Huggingface models compute the loss internally, the `Trainer` passes the input data to the model, extracts the loss from the output, and does the backward step.
It also handles additional steps to monitor the training process, like saving checkpoints of the model or logging the loss and other validation metrics.
A significant advantage of using the `Trainer` is its ability to do multi-device training without requiring the user to care about dispatching the models and data to multiple accelerators.
Also, it comes with an extension that allows more sophisticated tweaks, like training with 16bit-precision.

### Extending the `Trainer`

There are two different options to customize certain aspects of the behavior of the `Trainer`.
One option the customize the `Trainer` is the callback API
Callbacks are executed at certain events while the trainnig (e.g., at the end of an epoch) and they have access to many different things like the model or the `Trainer`.
But they are limited to read only operations which limits their scope to things like logging, saving certain parts of a model or stopping the training if a certain condition is met.

If further changes to the `Trainer` are required, the recommended way is to subclass it and create a custom version.
Internally, the `Trainer` structures the training into different substeps and exposes each of them via a method.
By overwriting these methods, it is possible to change certain parts of the logic without rewriting the rest of the code that would not be changed anyway. 
The most important methods to modify the `train-test-val`-loop itself are the `<train/test/val>-step` methods and the `compute_loss` method.
These methods implement the essential individual training steps and are used in all higher-order methods that handle the complete train, test, or validation loop.

### Logging

Logging is simple and is done automatically if a `logdir` is specified in the `TrainingArguments`. 
By default, it saves the logs to disk, using a Tensorboard-compliant format.
Additional logging steps, can be implemented by either overwriting the `.log`-method of Trainer or by using callbacks.
There are already some pre-built callbacks available to log the progress a text format or to log it using Weights and Biases.

#### Custom metrics

<!--Also a list of validation metrics can be computed by passing a function to the `Trainer` during initialization, and they automatically get logged too.
Another option to add custom metrics to the `Trainer` is to overwrite the -->


### Training Arguments

As stated above, a `TrainingArguments` object stores all hyperparameters of the training.
Storing all parameters in a single objects is useful to enable a proper reproducibility, since this object can easily be serialized and saved to disk.
Also, the `TrainingArguments` class works seamlessly with `Transformers` built-in parsing class, so the parameters can easily be made available through a command-line interface.

### HfArgumentParser

Most experiments are repeated several times with different parameters. Changing the hyperparameters directly in the code comes with several caveats.
Most importantly, it can harm reproducibility since tracking changes in the source code requires either version control and a strict commit regime or keeping several versions of the same file with different parameters. Also, it can be tedious the search for the location of all parameters across the code manually.
Making the hyperparameters adjustable via a command-line interface decouples the hyperparameters from the rest of the code and alleviates this issue.
While there are arguably a lot of different solutions to this problem with many strategies that are more sophisticated than a command-line interface, it is a good start. It has the advantage of being platform-independent without requiring additional dependencies. Also, it does not require learning additional tooling.

Huggingface comes with a built-in solution called `HfArgumentParser`. It creates command-line interfaces by exposing the fields of `dataclasses` as command-line arguments. 
Since many classes of the `transformers` library have corresponding configuration `dataclasses` that store all their parameters, the `HfArgumentParser` can be used to control nearly every aspect of the training. The example below shows how easy it is to create custom dataclasses, so that 

```{code-cell} ipython
from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class TrainArgs:
    batch_size: int = field(
        default = 8,
        metadata = {"help": "Number of batched for training."}
    )

parser = HfArgumentParser(TrainArgs)
parser.print_help()
train_args = parser.parse_args_into_dataclasses(["--batch_size", "4"])
print(train_args)
```

## Implementation

To use a custom loss function the developers recommend creating a custom class by inheriting from the `Trainer` and overwriting the `.compute_loss` method.

Another possibility would be to create a custom model with a sentence ordering head, that just like the other Huggingface models, computes the loss internally when called with labels. However, creating a custom model requires more work since each model has to come with a configuration object and follow strict guidelines to be interoperable with other Huggingface code.

Because of this, we choose the more straightforward solution and create a custom `SentenceOrderingTrainer.`

### Loss function

```python
class SentenceOrderingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.target_token_id = kwargs.pop("target_token_id")
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        # Get sentence indices
        batch_labels = inputs.pop("labels")

        # Get logits from model
        outputs = model(**inputs)
        batch_logits = outputs["logits"]

        # Get logits for all cls tokens
        batch_input_ids = inputs["input_ids"]

        loss_fn = nn.MSELoss(reduction="sum")
        batch_loss = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        
        for labels, logits, input_ids in zip(
            batch_labels, batch_logits, batch_input_ids
        ):

            true_labels = labels[labels != -100].reshape(-1)
            targets = true_labels.float()

            target_logits = logits[input_ids == self.target_token_id].reshape(-1)

            if target_logits.size(0) < targets.size(0):
                targets = targets[: target_logits.size(0)]

            batch_loss = batch_loss + loss_fn(targets, target_logits)

        loss = batch_loss / batch_logits.size(0)

        outputs["loss"] = loss
        return (loss, outputs) if return_outputs else loss
```

The implementation of the `.compute_loss` method is straightforward.
It receives the model and the input data as inputs.
This way, all data of the current batch is available, which is especially helpful in cases like ours where we need to check the input to compute the loss.
In addition, to our custom loss function, we also add another attribute to the `Trainer`, which holds the id of the target sentence token.
We leave the rest of the `Trainer` untouched.

### Metrics

Computing custom metrics while training does not require overwriting methods.
Instead, we can initialize the `Trainer` with a function that computes all additional metrics.

```python
def make_compute_metrics_func(target_token_id) -> Callable:
    def compute_ranking_func(eval_prediction: EvalPrediction) -> Dict[str, float]:
        batch_sent_idx, batch_input_ids = eval_prediction.label_ids
        batch_logits = eval_prediction.predictions.squeeze(2)

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
            predicted_idx = np.argsort(np.argsort(target_logits))
            tau, pvalue = kendalltau(sent_idx, predicted_idx)
            metrics["kendalls_tau"].append(tau)
            metrics["acc"].append(accuracy_score(sent_idx, predicted_idx))
            metrics["mean_logits"].append(logits.mean())
            metrics["std_logits"].append(logits.std())
        metrics = {metric: np.mean(scores) for metric, scores in metrics.items()}
        return metrics

    return compute_ranking_func
```

In contrast to the `.compute_loss`-method, which receives the input, this method only receives an EvalPrediciton object, a dictionary containing the model's outputs, and the labels from the dataset.
However, similar to the loss function, computing the metrics requires access to the input data to retrieve the indices of the target tokens. To control the content of an `EvalPrediction`, object we can use the `label_names` parameter of the `TrainingArguments`. It receives a list of keys from the input. These keys then get copied to the `EvalPrediction` objects.

```python
training_args = TrainingArguments(
	...,
	label_names=["labels", "input_ids"],
	...,
)
```
A minor but valuable trait of the `EvalPrediction` objects is that their content gets converted from `torch.tensors` to `np.arrays`. Because most predefined validation metrics use `Numpy`, this saves some manual conversions.


## Complete code

When we plug everything together, the code of our experiment looks like this:

```python
from transformers import TrainingArguments, HfArgumentParser
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer
from transformers import set_seed
from datasets import load_from_disk

from model import (
    SentenceOrderingTrainer,
    so_data_collator,
    make_compute_metrics_func,
    ModelArgs,
    make_tokenization_func,
)


if __name__ == "__main__":

    args_parser = HfArgumentParser((ModelArgs, TrainingArguments))
    model_args, training_args = args_parser.parse_args_into_dataclasses()

    # Add fixed args
    training_args.label_names = ["labels", "input_ids"]

    set_seed(training_args.seed)

    dataset = load_from_disk(
        "/home/keller/Uni/trf_training_tut/scripts/data/rocstories"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

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

    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, num_labels=1
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path, config=model_config
    )

    tokenization = make_tokenization_func(
        tokenizer=tokenizer,
        text_column="text",
        padding="max_length",
        truncation=True,
        add_special_tokens=False,
    )
    dataset = dataset.map(tokenization, batched=True)

    dataset = dataset.rename_column("so_targets", "labels")

    dataset.set_format("torch")

    metrics_func = make_compute_metrics_func(tokenizer.cls_token_id)

    trainer = SentenceOrderingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        target_token_id=tokenizer.cls_token_id,
        data_collator=so_data_collator,
        compute_metrics=metrics_func,
    )

    trainer.train()

    trainer.save_model(model_args.final_checkpoint_path)

```


## Conclusion

The Huggingface `Trainer` is a good choice when it comes to training models on standard tasks that are well supported.
In these cases, it enables to train models effortlessly without requiring to write much code.
It has a lot of useful out-of-the-box features, like gradient clipping, half-precision training, support of distributed training or logging to Tensorboard, which make it feasible for training large models on large datasets. In the best case, when the dataset is already available as Huggingface `Dataset` and the model task is also supported by the `transformers` library, it comes down to a few lines of code to train the model without having to really dive deep into any internals along the way.

Nonetheless, there are a few issues if one wants to leave the carved-out paths.
Like the rest of Huggingface's software, the' transformers' library is relatively young and fastly evolving. 
Huggingface's self-proclaimed goal is to provide an easy-to-use all-in-one infrastructure for NLP with language models and incorporate new models, architectures, and developments as quickly as possible.
On this path, sacrifices have to be made.

One area that seems to suffer from the speedy development is documentation.
It is sufficient and provides all essential information, but it can be very sparse in detail at times. 
Often, there are multiple options to choose from when customizing something. 
For example, the default optimizer can be exchanged during the initialization of the `Trainer`, by simply passing another one to it.
Another possibility would be to create a custom `Trainer` and to overwrite the `.create_optimizer`-method.
In these cases, the documentation lacks hints to decide which way to go.
Other times the documentation does not paint the whole picture of the behavior of the described object.
In these cases, it might become necessary to take a look into the source code itself.

There it becomes evident that the `Trainer` could use some refactoring.
Especially, its high-level methods, like the `.train`-method, are very complex since they do much heavy lifting, for example, dispatching the training to multiple devices.
While the preferred way to customize the training is to subclass the `Trainer` and overwrite methods, this is only feasible for the low-level methods that define single steps. Even tiny adjustments to the high-level methods can require copying code or rewriting certain parts.