import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from pathlib import Path
from helpers.filehandling import create_file
from typing import Tuple

BASE_PATH = Path(__file__).parent
TEMP_PATH = BASE_PATH.joinpath(Path("./temp"))
SAMPLE_PATH = TEMP_PATH.joinpath(Path("./sampled"))

TEXT_TRAIN = SAMPLE_PATH.joinpath(Path("./text_train.pickle"))
TEXT_TEST = SAMPLE_PATH.joinpath(Path("./text_test.pickle"))
LABELS_TRAIN = SAMPLE_PATH.joinpath(Path("./labels_train.pickle"))
LABELS_TEST = SAMPLE_PATH.joinpath(Path("./labels_test.pickle"))

FILE_PATHS = [
    TEXT_TRAIN,
    TEXT_TEST,
    LABELS_TRAIN,
    LABELS_TEST,
]


def split_dataset(
    text: pd.DataFrame,
    labels: pd.DataFrame,
    test_size: float,
    stratify: pd.DataFrame,
    random_state: int,
):
    files = train_test_split(
        text,
        labels,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    for file, path in zip(files, FILE_PATHS):
        with create_file(path, mode="wb", encoding=None, buffering=-1) as f:
            f.write(pickle.dumps(file))


def load_split_dataset(
    file: Path,
) -> pd.DataFrame:
    with open(file, mode="rb") as pickle_file:
        return pickle.load(pickle_file)
