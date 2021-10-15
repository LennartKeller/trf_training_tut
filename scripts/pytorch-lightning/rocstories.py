from argparse import ArgumentParser

from datasets import load_from_disk
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import AutoTokenizer

from pl_modules import (
    HuggingfaceDatasetWrapper,
    PlLanguageModelForSequenceOrdering,
    so_data_collator,
)


def main(hparams):
    seed_everything(hparams.seed)
    print("Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(hparams.model_name_or_path)

    print("Loading datasets.")
    data = load_from_disk("../data/rocstories")

    # Downsampling for debugging...
    data = data.filter(lambda _, index: index < 5000, with_indices=True)

    dataset = HuggingfaceDatasetWrapper(
        data,
        text_column="text",
        target_column="so_targets",
        tokenizer=tokenizer,
        mapping_funcs=[],
        collate_fn=so_data_collator,
        train_batch_size=hparams.train_batch_size,
        eval_batch_size=hparams.val_batch_size,
    )

    if tokenizer.cls_token != "[CLS]":
        print("Non BERT-Model. Updating special sentence token.")

        def replace_special_token(entry, sent_token=tokenizer.sep_token):
            for idx, text in enumerate(entry["text"]):
                text = text.replace("[CLS]", sent_token)
                print(text)
                entry["text"][idx] = text
            return entry

        dataset.map(replace_special_token, batched=True)

        hparams.target_token_id = tokenizer.sep_token_id
        print(f"Updated special token id: {hparams.target_token_id}")

    print("Loading model.")
    model = PlLanguageModelForSequenceOrdering(hparams=hparams)

    print("Initializing trainer.")
    # Init loggers
    loggers = []
    tensorboard_logger = TensorBoardLogger(save_dir=hparams.default_root_dir + "/logs")
    loggers.append(tensorboard_logger)

    # Init callbacks
    callbacks = []
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", save_top_k=3, mode="min")
    callbacks.append(checkpoint_callback)

    trainer = Trainer.from_argparse_args(hparams, logger=loggers, callbacks=callbacks)

    print("Start tuning.")
    trainer.tune(model=model, datamodule=dataset)

    print("Start training.")
    trainer.fit(model=model, datamodule=dataset)

    print("Start testing.")
    trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser = Trainer.add_argparse_args(parser)
    parser = PlLanguageModelForSequenceOrdering.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
