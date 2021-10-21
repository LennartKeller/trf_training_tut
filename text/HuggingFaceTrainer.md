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

## Design & Philophy
 
The most natural pick for training a Huggingface model is the `Trainer` that ships with the `transformers` library itself.
It is designed to handle the training of Huggingface models in "most standard cases."

Naturally, the `Trainer` is designed to work in the Huggingface ecosystem. It receives the data as Huggingface Dataset and expects the models to follow the conventions of the `transformers` library.
In theory, other custom models could be used with the `Trainer` too, but the developers warn that this might lead to strange side effects.


## Classes

__Trainer__

Unsurprisingly, the `Trainer` class handles the training. 
Its configuration is outsourced to a `TrainingArguments` class that stores all relevant parameters for training.
These arguments are passed to the `Trainer`  alongside a model and a dataset during initialization.
The training then starts automatically. Since Huggingface models compute the loss internally, in most standard cases, the `Trainer`  passes the input data to the model, extracts the loss from the output, and does the backward step. It also has built-in logging capabilities to log the loss and other metrics.
Another main advantage of the `Trainer` lies in its ability to do multi-device training without requiring the user to care of dispatching the model and data to all devices manually. Also, it comes with an extension that allows more sophisticated tweaks, like training with half-precision.

By design, the `Trainer` is one monolithic block that handles the training end-to-end without relying internally on subclasses that structure the process.

There are two different options to customize certain aspects of the training.

Firstly, callbacks can be created using a dedicated API.
These callbacks are then executed at certain events while training (i.e., the end of an epoch).
The main purpose of callbacks is not to modify but to extend the `Trainer`.
Callbacks are an easy tool to define additional operations, like pruning the model after each epoch or modifying the gradients before each update-step of the optimizer.

Secondly, if a task requires modifying the `train-test-val`-loop itself, creating a custom trainer via subclassing is the best option.
Each of these loops is structured in a two-fold way. A method that ends with `<train/test/val>-step` defines the logic to process a single batch of data, while the `<train/test/val>-step` methods define the loops as a whole. So it is possible to device whether to change parts of these stages or to rewrite them completely.

__Training Arguments__

As stated above, the hyperparameters of training are passed to the trainer as a `TrainingArguments` object. One of the advantages is that this class can easily be serialized and saved to different formats. Also, it works seamlessly with `Transformers` built-in parsing class, so the parameters can easily be made available through a command-line interface.


## Features
Furthermore, the `Trainer` class not only handles the training loop but comes with additional features.
Most notably, it supports multi-device training out of the box.
Another essential aspect of machine learning experiments is logging the progress during training and saving checkpoints for further evaluation.
The `Trainer` class is capable of doing both.
By default, it logs the progress to the standard output of the current runtime environment (e.g., command line or notebook). 

If additionally, a `logdir` is specified, it also saves the logs to disk, using a Tensorboard-compliant format.

Validation metrics can be computed by passing a function to the trainer during initialization, and they automatically get logged, using one (or more) of its logging modules.
Even more exotic features, like training with half-precision, are supported.


## Implementation

To use a custom loss function the developers recommend creating a custom class by inheriting from the `Trainer` and overwriting the `.compute_loss` method.

Another possibility would be to create a custom model with a sentence ordering head, that just like the other Huggingface models, computes the loss internally when called with labels. However, creating a custom model requires more work since each model has to come with a configuration object and follow strict guidelines to be interoperable with other Huggingface code.

Because of this, we choose the more straightforward solution and create a custom `SentenceOrderingTrainer.`

__Loss function__

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
In addition, to our custom loss function, we also add another attribute to the `Trainer`, which holdS the id of the target sentence token.
We leave the rest of the `Trainer` untouched.

__Metrics__

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
A minor but useful trait of the `EvalPrediction` objecs that their content get converted from `torch.tensors` to `np.arrays`. Because most predefined validation metrics use `Numpy`, this saves some lines of code.

__HfArgumentParser__

Most experiments are repeated several times with different parameters. Changing the hyperparameters directly in the code comes with several caveats. Most importantly, it can harm reproducibility since tracking changes in the source code requires either version control and a strict commit regime or keeping several versions of the same file with different parameters. Also, it can be tedious the search for the location of all parameters across the code manually.
Making the hyperparameters adjustable via a command-line interface decouples the hyperparameters from the rest of the code and alleviates this issue.
While there are arguably a lot of different solutions to this problem with many strategies that are more sophisticated than a command-line interface, it is a good start. It has the advantage of being platform-independent without requiring additional dependencies. Also, it does not require learning additional tooling.

Huggingface comes with a built-in solution called `HfArgumentParser`. It creates command-line interfaces by exposing the fields of `dataclasses` to command-line arguments. 
Since `dataclasses` are widely used in the Huggingface ecosystem to define configurations (i.e., TrainingArguments), it can be used to controll the configuration of nearly each aspect of the training. Only the 

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
# In the real world the .parse_args... method
# would be called without any arguments
# to parse data from the commandline.
train_args = parser.parse_args_into_dataclasses(["--batch_size", "4"])
print(train_args)
```

### Complete code

When everything is plugged together the experiment itself, looks like this:


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
This script is fully customaizable via command-line arguments.
Note, that the trainer itself and all other helper functions are outsourced to a extern file and only imported for the sake of readability.

## Conclusion

__Pros__

__Contras__

* Undocumented: You have to look into the code quite often
* Specifically tailord to exstiing huggingface models and workflows