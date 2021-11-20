import json
from poutyne.framework import experiment
from torch.optim import AdamW
from poutyne import (
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

    MODEL_NAME_OR_PATH = "distilbert-base-cased"
    LEARNING_RATE = 3e-5
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 16
    DEVICE = 0
    N_EPOCHS = 3
    SAVE_DIR = "experiments/rocstories/distilbert"

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

    writer = SummaryWriter("runs/distilbert/1")
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
    )

    experiment.train(
        train_generator=train_dataloader,
        valid_generator=val_dataloader,
        epochs=N_EPOCHS,
        save_every_epoch=True,
    )

    test_results = experiment.test(test_generator=test_dataloader)
    with open(f"test_results_{MODEL_NAME_OR_PATH}.json", "w") as f:
        json.dump(test_results, f)
