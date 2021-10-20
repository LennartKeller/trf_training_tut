from transformers import Trainer, TrainingArguments, HfArgumentParser
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass, field

@dataclass
class ModelArgs:
    model_name_or_path: str = field(
        default="bert-base-cased", metadata={"help": "Path to pretrained model or model or id in huggingface hub."}
    )


if __name__ == "__main__":

    args_parser = HfArgumentParser((ModelArgs, TrainingArguments))
    model_args, training_args = args_parser.parse_args_into_dataclasses()
    print(model_args)
    print(training_args)
    ####################
    model = AutoModel.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    Trainer(model=model, tokenizer=tokenizer, args=training_args)
