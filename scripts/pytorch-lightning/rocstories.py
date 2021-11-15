from os.path import basename
from datasets import load_from_disk
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningArgumentParser
from transformers import AutoTokenizer

from pl_modules import (
    HuggingfaceDatasetWrapper,
    PlLanguageModelForSequenceOrdering,
    so_data_collator,
)


def main(model_args, trainer_args, checkpoint_args, tensorboard_args, run_args):

    seed_everything(run_args["seed"])

    print("Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(model_args["model_name_or_path"])

    print("Loading datasets.")
    data = load_from_disk("../data/rocstories")

    # Downsampling for debugging...
    # data = data.filter(lambda _, index: index < 5000, with_indices=True)

    dataset = HuggingfaceDatasetWrapper(
        data,
        text_column="text",
        target_column="so_targets",
        tokenizer=tokenizer,
        mapping_funcs=[],
        collate_fn=so_data_collator,
        train_batch_size=run_args["train_batch_size"],
        eval_batch_size=run_args["val_batch_size"],
    )

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
        model_args["target_token_id"] = tokenizer.cls_token_id

    print("Loading model.")
    model = PlLanguageModelForSequenceOrdering(hparams=model_args)

    print("Initializing trainer.")
    # Init logger
    tensorboard_logger = TensorBoardLogger(**tensorboard_args)

    # Init callbacks
    callbacks = []
    checkpoint_callback = ModelCheckpoint(**checkpoint_args)
    callbacks.append(checkpoint_callback)

    # Remove default args
    trainer_args.pop("logger")
    trainer_args.pop("callbacks")
    trainer = Trainer(logger=tensorboard_logger, callbacks=callbacks, **trainer_args)

    print("Start tuning.")
    trainer.tune(model=model, datamodule=dataset)

    print("Start training.")
    trainer.fit(model=model, datamodule=dataset)

    print("Start testing.")
    test_results = trainer.test()
    print(test_results)



if __name__ == "__main__":
    parser = LightningArgumentParser()
    group = parser.add_argument_group()
    group.add_argument("--run.run_name", type=str, default=basename(__file__))
    group.add_argument("--run.seed", type=int, default=0)
    group.add_argument("--run.train_batch_size", type=int, default=8)
    group.add_argument("--run.val_batch_size", type=int, default=16)

    parser.add_lightning_class_args(ModelCheckpoint, "checkpoint")
    parser.add_class_arguments(TensorBoardLogger, nested_key="tensorboard")
    parser.add_lightning_class_args(Trainer, "trainer")
    parser = PlLanguageModelForSequenceOrdering.add_model_specific_args(parser)

    args = parser.parse_args()

    model_args = args.get("model", {})
    trainer_args = args.get("trainer", {})
    checkpoint_args = args.get("checkpoint", {})
    tensorboard_args = args.get("tensorboard", {})
    run_args = args.get("run", {})

    main(model_args, trainer_args, checkpoint_args, tensorboard_args, run_args)
