from argparse import ArgumentParser

from datasets import load_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pl_modules import (
    HuggingfaceDatasetWrapper,
    PlAutoModelForSequenceClassification,
    HuggingfaceTrainer,
    HugginfaceWrapper,
)


def main(hparams):
    print("Loading model & tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-cased")
    text_classificator = HugginfaceWrapper(model=model)

    print("Loading datasets.")
    # Downsampling for debugging...
    data = load_dataset("imdb")
    data["train"] = data["train"].train_test_split(train_size=0.001, seed=42)["train"]
    data["test"] = data["test"].train_test_split(train_size=0.001, seed=42)["train"]
    dataset = HuggingfaceDatasetWrapper(
        data, text_column="text", target_column="label", tokenizer=tokenizer
    )

    print("Initializing trainer.")
    trainer = Trainer.from_argparse_args(hparams)

    print("Start training.")
    trainer.fit(model=text_classificator, datamodule=dataset)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
