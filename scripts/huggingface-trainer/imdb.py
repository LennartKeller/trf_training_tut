from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer
from datasets.load import load_dataset
from transformers.training_args import TrainingArguments


def init_model(*args)

if __name__ == '__main__':
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-cased')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased')

    dataset = load_dataset('imdb')

    dataset = dataset.map(lambda entry: tokenizer(entry['text'], padding=True, truncation=True))
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    dataset = dataset.rename_column('label', 'labels')

    trainer = Trainer(mode,)


