import pickle
from pathlib import Path
from typing import cast
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
from data import create_file
from data import sampling_filepath
from data import SAMPLING_TEMP
from sklearn.model_selection import train_test_split


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
) -> List[Path]:
    partial_files = train_test_split(
        text,
        labels,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )
    paths: List[Path] = []

    for partial_file, filename in zip(partial_files, FILES):
        filepath = sampling_filepath(name=name, filename=filename)
        with create_file(
            file_path=filepath,
            mode="wb",
            encoding=None,
            buffering=-1,
        ) as f:
            f.write(pickle.dumps(partial_file))
            paths.append(filepath)
    return paths


def load_split_dataset(
    name: str,
    filename: str,
) -> pd.DataFrame:
    with open(
        sampling_filepath(name=name, filename=filename), mode="rb"
    ) as pickle_file:
        return cast(pd.DataFrame, pickle.load(pickle_file))
