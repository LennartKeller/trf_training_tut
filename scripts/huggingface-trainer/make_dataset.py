import pandas as pd
import numpy as np
from more_itertools import flatten

from datasets import Dataset, DatasetDict, ClassLabel, Sequence


def make_classlabel_encoder(column_name, class_labels):
    """
    Returns a function which takes an row from a huggingface dataset as inputs an converts the column with passed name
    from a string label to int label using the the passed ClassLabel object.
    """

    def encoder(entry):
        return {column_name: class_labels.str2int(entry[column_name])}

    return encoder


def make_multilabel_encoder(column_name, class_labels):
    """
    Works similar to the make_classlabel_encoder function but returns a binary vector indicating multilabel assignments.
    """
    n_labels = len(class_labels.names)

    def encoder(entry):
        binarized = np.zeros(n_labels, dtype="int8")
        label_idc = [class_labels.str2int(label) for label in entry[column_name]]
        binarized[label_idc] = 1
        return {column_name: binarized}

    return encoder


if __name__ == "__main__":

    df_train = pd.read_json("../data/imdb_train.json")
    df_val = pd.read_json("../data/imdb_val.json")
    df_test = pd.read_json("../data/imdb_test.json")
    print(df_train.shape, df_val.shape, df_test.shape)

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(df_train, split="train"),
            "val": Dataset.from_pandas(df_val, split="train"),
            "test": Dataset.from_pandas(df_test, split="test"),
        }
    )

    # Convert titleType column
    uniq_title_type_labels = list(set(dataset["train"]["titleType"]))
    title_type_labels = ClassLabel(
        num_classes=len(uniq_title_type_labels), names=uniq_title_type_labels
    )

    title_type_encoder = make_classlabel_encoder("titleType", title_type_labels)
    dataset = dataset.map(title_type_encoder, batched=True)

    updated_features = dataset["train"].features.copy()
    updated_features["titleType"] = title_type_labels
    dataset = dataset.cast(updated_features)

    # Convert genre column

    uniq_genres = list(set(flatten(dataset["train"]["genre"])))
    genre_labels = ClassLabel(num_classes=len(uniq_genres), names=uniq_genres)

    genre_labels_encoder = make_multilabel_encoder("genre", genre_labels)
    dataset = dataset.map(genre_labels_encoder)  # batched mode does not work here ...

    updated_features = dataset["train"].features.copy()
    genre_column = Sequence(feature=genre_labels)
    updated_features["genre"] = genre_column
    dataset = dataset.cast(updated_features)

    # Save dataset to disk
    dataset.save_to_disk("../data/imdb_huggingface")
