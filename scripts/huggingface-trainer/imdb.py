from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers import BertTokenizerFast
from transformers import set_seed
from model import BertForMultiHeadConfig, BertForMultiHeadModel
from datasets import load_from_disk


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

if __name__ == '__main__':
    set_seed(42)

    dataset = load_from_disk('../data/imdb_huggingface')
    
    num_classes = dataset['train'].features['titleType'].num_classes
    num_labels = dataset['train'].features['genre'].feature.num_classes

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased')
    model_config = BertForMultiHeadConfig.from_pretrained('bert-base-german-cased', num_labels=num_classes, num_multi_labels=num_labels)
    model = BertForMultiHeadModel.from_pretrained('bert-base-german-cased', config=model_config)

    tokenization = make_tokenization_func(tokenizer=tokenizer, text_column='text', padding=True, truncation=True)
    dataset = dataset.map(tokenization, batched=True)
    
    rename_target_columns = make_rename_func({'titleType': 'labels', 'genre': 'multi_labels', 'averageRating': 'regression_targets'})
    dataset = dataset.map(rename_target_columns, batched=True)

    dataset.set_format('torch')

    training_args = TrainingArguments(
        output_dir='test',
        overwrite_output_dir=True,
        learning_rate=10e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        evaluation_strategy='steps',
        eval_steps=200,
        num_train_epochs=5,
        logging_dir='test/logs',
        logging_steps=200,
        save_strategy='steps',
        save_steps=2000,
        remove_unused_columns=True
        )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['val']
    )

    trainer.train()
    
    trainer.save_model('test/model')







