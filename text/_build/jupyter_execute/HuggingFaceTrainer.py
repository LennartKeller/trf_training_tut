#!/usr/bin/env python
# coding: utf-8

# # Huggingface Trainer
# 
# Since Huggingface proclaimed goal is to provide an environment to develop and train all sorts of language models, they also ship a solution for training models.
# It is called the `Trainer`, and is integrated into the `transformers` library itself.
# Of course, it is profoundly integrated into the Huggingface-ecosystem and can train most `transformers` models out of the box.
# 
# ## Classes
# 
# ### Trainer
# 
# Design-wise, the `Trainer` is one single class that handles the training end-to-end.
# Its configuration is outsourced to a `TrainingArguments` class that stores all relevant parameters for training.
# These arguments are passed to the `Trainer`  alongside a model and a dataset during initialization.
# Since Huggingface models compute the loss internally, the `Trainer` passes the input data to the model, extracts the loss from the output, and does the backward step.
# It also handles additional steps to monitor the training process, like saving checkpoints of the model or logging the loss and other validation metrics.
# A significant advantage of using the `Trainer` is its ability to do multi-device training without requiring the user to care about dispatching the models and data to multiple accelerators.
# Also, it comes with an extension that allows more sophisticated tweaks, like training with 16bit-precision.
# 
# #### Extending the `Trainer`
# 
# There are two different options to customize certain aspects of the behavior of the `Trainer`.
# Additional read-only operations can be implemented with the callback API.
# Callbacks are executed at specific events during the training (e.g., at the end of an epoch).
# They have access to many different things like the model or the current state of the `Trainer`.
# However, since they can not manipulate their environment, their scope is limited to logging, saving certain parts of a model, or stopping the training if a specific condition is met.
# 
# If further changes to the `Trainer` are required, the recommended way is to subclass it and create a custom via inheritance.
# Internally, the `Trainer` structures the training into different sub-steps and exposes them via a method for each of them.
# By overwriting these methods, it is possible to change certain parts of the logic without rewriting the rest of the code that would not be changed anyway.
# The most important methods to modify the `train-test-val`-loop itself are the `<train/test/val>-step` methods and the `compute_loss` method.
# These methods implement the essential individual training steps and are called within methods that implement higher-order operations like the `.train`-method, which handles the complete training loop.
# 
# 
# #### Logging
# 
# If a `logdir`-argument is specified in the `TrainingArguments`-object, logging is enabled automatically.
# By default, the `Trainer` outputs the logs in two formats: Stdout and disk, using a Tensorboard-compliant format.
# Additional logging can be implemented by either overwriting the `.log`-method of Trainer or by using callbacks.
# There are already some pre-built callbacks available. For example, to log the progress to Weights and Biases or a CSV table.
# 
# #### Custom metrics
# 
# Since the `Trainer` is agnostic towards the task it is used with; it only logs the loss by default.
# Additionally, metrics can be added by equipping the `Trainer` with a function that computes them during initialization.
# This function receives an `EvalPrediction` object.
# This object holds all predictions of the model and the valid labels.
# The output of the custom metric function ought to be a dictionary containing the name of the metric as key and the score as value.
# 
# ### Training Arguments
# 
# As stated above, a `TrainingArguments` object stores all hyperparameters of the training.
# Storing all parameters in a single object is helpful to ensuring reproducibility since this object can easily be serialized and saved to disk as JSON using its `.to_json_string`-method.
# Also, the `TrainingArguments` class works seamlessly with the built-in CLI-parser class of `transformers`, which helps make the configuration of an experiment available through a command-line interface.
# 
# ### HfArgumentParser
# 
# Most experiments are repeated several times with different parameters. These parameters have to be changed directly in the source code by default, which is not ideal for several reasons.
# Most importantly, it can harm reproducibility since tracking changes in the source code requires either version control and a strict commit regime or keeping several versions of the same file with different parameters. Also, it can be tedious the search for the location of all parameters across the code manually.
# Making the hyperparameters adjustable via a command-line interface decouples their configuration from the rest of the code, alleviating this issue.
# While there are arguably a lot of different solutions to this problem with many strategies that are more sophisticated than a command-line interface, it is an excellent first step.
# Moreover, it has the advantage of being platform-independent without depending on additional dependencies.
# 
# Huggingface provides a built-in solution for building these interfaces called `HfArgumentParser`.
# It is an extended version of Pythons `argsparse` parser and creates command-line interfaces by parsing the fields of `dataclasses` and exposing them as command-line arguments.
# Since most configuration classes of the `transformers` library are `dataclasses,` the `HfArgumentParser` can flexibly control nearly every aspect of the training.
# Further extending the arguments can be easily done by creating custom `dataclasses` that hold additional parameters.

# In[1]:


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


# ## Implementation
# 
# ### Loss function
# 
# For the sentence ordering task, we employ a language model with a standard token-classification head.
# However, since the task requires a custom loss function, we have to discard the loss of the model and use our custom loss function.
# To do so, we follow the guidelines and create our custom version of the `Trainer` with a custom `.compute_loss` function.
# The implementation <!--of the `.compute_loss` method --> is straightforward.
# The `.compute_loss` method receives a reference to the model and the input data as inputs, which is especially helpful in cases like ours where we need to check the `input_ids` to compute the loss.
# In addition, to our custom loss function, we also add another attribute to the `Trainer`, which holds the id of the target sentence token in order to find the correct tokens in the input sequence.
# We leave the rest of the `Trainer` untouched.

# In[ ]:


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


# ### Metrics
# 
# To compute custom metrics during validation, we need to create a function.
# The function computes all metrics at once.
# In contrast to the `.compute_loss`-method, which receives the input and the model, it receives an `EvalPrediction` object as input.
# An `EvalPrediction` contains the model's outputs and the labels from the dataset.
# However, similar to the loss function, computing the metrics requires access to the input data to retrieve the indices of the target tokens.
# To control the content of an `EvalPrediction` object, we can use the `label_names` parameter of the `TrainingArguments`.
# With this argument, we can specify additional fields that are copied from the input batches to the `EvalPrediction` objects.
# This way, we can incorporate the labels and the `input_ids` of tokens in the `EvalPrediction` object.
# 
# A minor but valuable trait of the `EvalPrediction` objects is that their content gets converted from `torch.tensors` to `np.arrays`.
# Because most validation metrics from other libraries use NumPy, we do not need to convert the data manually.

# In[ ]:


training_args = TrainingArguments(
    ...,
    label_names=["labels", "input_ids"],
    ...,
)


# In[ ]:


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


# ### Custom CLI arguments
# 
# We use the `HfArgumentParser` to make the parameters of our experiment adjustable via the command line.
# In addition to the `TrainingsArguments`, we also want to control the type of the model.
# Custom parameters can easily be added by creating a custom `dataclass`. 
# We create a `ModelArgs` class that has two fields. One to specify the name or path to the model and a second parameter to specify the path where the final model is saved after training.

# In[ ]:


from dataclasses import dataclass, field
from transformers import TrainingArguments, HfArgumentParser

@dataclass
class ModelArgs:
    model_name_or_path: str = field(
        default="bert-base-cased",
        metadata={
            "help": "Path to pretrained model or model or its name to load it from Huggingface Hub."
        },
    )

    final_checkpoint_path: str = field(
        default=None, metadata={"help": "Path to save the final model."}
    )

...

args_parser = HfArgumentParser((ModelArgs, TrainingArguments))
model_args, training_args = args_parser.parse_args_into_dataclasses()


# ## Complete code
# 
# After moving our custom code for the `Trainer` and the metric function to an external module, the rest of the code to implement the experiment looks like Listing (TODO).
# There are only two steps left to complete the script.
# Firstly, we must ensure that our data always contains the correct special tokens for ordering the sentences.
# Since we prepared the data beforehand by adding BERTs special `[SEP]`-token as a prefix to each sentence, we have to ensure that these tokens are replaced if necessary using the  `replace_cls_token` function.
# 
# Lastly, we want to control the randomness in our experiment to make it consistently reproducible.
# The `transformers` library comes with a helpful function called `set_seed`, which controls the state of all random number generators of Python itself, NumPy, and PyTorch at once.

# In[ ]:


import json
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

    test_results = trainer.evaluate(eval_dataset=dataset["test"])
    with open(f"test_results_{model_args.model_name_or_path}.json", "w") as f:
        json.dump(test_results, f)

    print(test_results)


# ## Conclusion
# 
# The Huggingface `Trainer` is a perfect choice when training models on standard tasks that are well supported.
# In these cases, it enables to train models effortlessly without requiring to write much code.
# In the best case, when the dataset is already available as Huggingface `Dataset`, it comes down to a few lines of code to train the model without having to dive deep into any internals along the way.
# 
# Also, it has many useful out-of-the-box features, like gradient clipping, half-precision training, support of distributed training, or logging to Tensorboard, which make it feasible for training large models on large datasets.
# 
# Nonetheless, there are a few issues if one wants to leave the carved-out paths.
# Like the rest of Huggingface's software, the `transformers` library is relatively new and evolves at great speed.
# Huggingface's self-proclaimed goal is to provide an easy-to-use all-in-one infrastructure for NLP with language models and incorporate new models, architectures, and developments as quickly as possible.
# On this path, sacrifices have to be made.
# 
# One area that seems to suffer from the speedy development is documentation.
# It is sufficient and provides all essential information, but it can sometimes be very sparse in detail.
# Often, there are multiple options to choose from when customizing something.
# For example, the default optimizer can be exchanged during the initialization of the `Trainer`, by simply passing another one to it. Or by overwriting the `.create_optimizer`-method.
# In cases like this one, the documentation lacks hints to decide which way to go.
# 
# Other times the documentation does not paint the whole picture of the behavior of the described object.
# In these cases, it might become necessary to take a look into the source code itself.
# 
# By looking into the source code of `Trainer`, it becomes evident that it could use some refactoring.
# Especially, its high-level methods, like the `.train`-method, are very complex since they do much heavy lifting, for example, dispatching the training to multiple devices.
# While the preferred way to customize the training is to subclass the `Trainer` and overwrite methods, this is only feasible for the low-level methods that define single steps. Even tiny adjustments to the high-level methods can require copying code or rewriting certain parts.
