from transformers import TrainingArguments, HfArgumentParser
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer
from transformers import set_seed
from datasets import load_from_disk

from model import SentenceOrderingTrainer, so_data_collator, make_compute_metrics_func, ModelArgs


def make_tokenization_func(tokenizer, text_column, *args, **kwargs):
    def tokenization(entry):
        return tokenizer(entry[text_column], *args, **kwargs)

    return tokenization


def make_rename_func(mapping, remove_src=False):
    def rename(entry):
        for src, dst in mapping.items():
            if remove_src:
                data = entry.pop(src)
            else:
                data = entry[src]
            entry[dst] = data
        return entry

    return rename


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
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=1)
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

    rename_func = make_rename_func({"so_targets": "labels"})
    dataset = dataset.map(rename_func, batched=True)

    dataset.set_format("torch")

    metrics_func = make_compute_metrics_func(tokenizer.cls_token_id)

    # training_args = TrainingArguments(
    #     output_dir="checkpoints/rocstories",
    #     overwrite_output_dir=True,
    #     learning_rate=3e-5,
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=16,
    #     evaluation_strategy="steps",
    #     gradient_accumulation_steps=1,
    #     eval_steps=1000,
    #     num_train_epochs=3,
    #     logging_dir="logs/rocstories",
    #     logging_steps=200,
    #     save_strategy="steps",
    #     save_steps=10000,
    #     remove_unused_columns=True,
    #     logging_first_step=True,
    #     prediction_loss_only=False,
    #     label_names=["labels", "input_ids"],
    # )

    trainer = SentenceOrderingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],  # .select(list(range(300))),
        target_token_id=tokenizer.cls_token_id,
        data_collator=so_data_collator,
        compute_metrics=metrics_func,
    )

    trainer.train()

    trainer.save_model("final_models/rocstories")
