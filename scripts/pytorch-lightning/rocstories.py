from argparse import ArgumentParser

from datasets import load_from_disk
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from transformers import AutoModelForTokenClassification, AutoTokenizer

from pl_modules import (
    HuggingfaceDatasetWrapper,
    PlLanguageModelForSequenceOrdering,
    so_data_collator,
)


def main(hparams):
    print(hparams)
    print('Loading model & tokenizer.')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    transformers_model = AutoModelForTokenClassification.from_pretrained(
            'bert-base-cased', return_dict=True, num_labels=1
        )
    model = PlLanguageModelForSequenceOrdering(model=transformers_model, target_token_id=tokenizer.cls_token_id)

    print('Loading datasets.')
    # Downsampling for debugging...
    data = load_from_disk('../data/rocstories')
    dataset = HuggingfaceDatasetWrapper(
        data, text_column='text', target_column='so_targets', tokenizer=tokenizer, mapping_funcs=[], collate_fn=so_data_collator
    )

    print('Initializing trainer.')
    trainer = Trainer.from_argparse_args(hparams)

    print('Start training.')
    trainer.fit(model=model, datamodule=dataset)
    trainer.test()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    print(args)
    main(args)
