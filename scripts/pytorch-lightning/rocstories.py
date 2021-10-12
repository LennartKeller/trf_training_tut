from argparse import ArgumentParser

from datasets import load_from_disk
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from transformers import AutoTokenizer

from pl_modules import (
    HuggingfaceDatasetWrapper,
    PlLanguageModelForSequenceOrdering,
    so_data_collator,
)


def main(hparams):
    seed_everything(hparams.seed)
    print("Loading model & tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = PlLanguageModelForSequenceOrdering(hparams=hparams)
    print("Loading datasets.")

    data = load_from_disk("../data/rocstories")

    # Downsampling for debugging...
    # data = data.filter(lambda _, index: index < 100, with_indices=True)

    dataset = HuggingfaceDatasetWrapper(
        data,
        text_column="text",
        target_column="so_targets",
        tokenizer=tokenizer,
        mapping_funcs=[],
        collate_fn=so_data_collator,
    )

    print("Initializing trainer.")
    tensorboard_logger = TensorBoardLogger(save_dir="logs/")
    trainer = Trainer.from_argparse_args(hparams, logger=tensorboard_logger)

    print("Start tuning.")
    trainer.tune(model=model, datamodule=dataset)
    print("Start training.")
    trainer.fit(model=model, datamodule=dataset)
    print("Start testing.")
    trainer.test()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser = Trainer.add_argparse_args(parser)
    parser = PlLanguageModelForSequenceOrdering.add_model_specific_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
