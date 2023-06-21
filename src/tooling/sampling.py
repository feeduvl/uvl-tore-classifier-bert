import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from pathlib import Path
from data import create_file
from typing import Tuple, cast
from data import SAMPLING_TEMP, sampling_filepath


TEXT_TRAIN = "text_train.pickle"
TEXT_TEST = "text_test.pickle"
LABELS_TRAIN = "labels_train.pickle"
LABELS_TEST = "labels_test.pickle"

FILES = [
    TEXT_TRAIN,
    TEXT_TEST,
    LABELS_TRAIN,
    LABELS_TEST,
]


def split_dataset(
    name: str,
    text: pd.DataFrame,
    labels: pd.DataFrame,
    test_size: float,
    stratify: pd.DataFrame,
    random_state: int,
):
    partial_files = train_test_split(
        text,
        labels,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    for partial_file, filename in zip(partial_files, FILES):
        with create_file(
            sampling_filepath(name=name, filename=filename),
            mode="wb",
            encoding=None,
            buffering=-1,
        ) as f:
            f.write(pickle.dumps(partial_file))


def load_split_dataset(
    name: str,
    filename: str,
) -> pd.DataFrame:
    with open(
        sampling_filepath(name=name, filename=filename), mode="rb"
    ) as pickle_file:
        return cast(pd.DataFrame, pickle.load(pickle_file))
