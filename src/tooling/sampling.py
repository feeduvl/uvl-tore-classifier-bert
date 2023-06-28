import pickle
from pathlib import Path
from typing import cast
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import overload
from typing import Tuple

import numpy as np
import pandas as pd
from data import create_file
from data import sampling_filepath
from data import SAMPLING_TEMP
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from strictly_typed_pandas import DataSet
from tooling.model import DataDF

DataTrain = Literal["data_train"]
DATA_TRAIN = "data_train"

DataTest = Literal["data_test"]
DATA_TEST = "data_test"


FILES = (
    DATA_TRAIN,
    DATA_TEST,
)


def split_dataset_k_fold(
    name: str,
    dataset: pd.DataFrame,
    folds: int,
    random_state: int,
) -> Iterator[Tuple[int, List[Path]]]:
    data: DataSet[DataDF] = dataset[
        ["sentence_id", "sentence_idx", "string", "tore_label"]
    ]
    sentences = dataset["sentence_id"]

    splitter = GroupKFold(n_splits=folds)
    for iteration, (train_index, test_index) in enumerate(
        splitter.split(X=data, groups=sentences)
    ):
        data_train = data.loc[train_index]
        data_test = data.loc[test_index]

        partial_files = (data_train, data_test)

        paths: List[Path] = []

        for partial_file, base_filename in zip(partial_files, FILES):
            filename = f"{iteration}_{base_filename}"

            filepath = sampling_filepath(
                name=name, filename=filename + ".pickle"
            )
            with create_file(
                file_path=filepath,
                mode="wb",
                encoding=None,
                buffering=-1,
            ) as f:
                partial_file.to_pickle(f)
                paths.append(filepath)

            filepath = sampling_filepath(name=name, filename=filename + ".csv")
            with create_file(
                file_path=filepath,
                mode="wb",
                encoding=None,
                buffering=-1,
            ) as f:
                partial_file.to_csv(f)
                paths.append(filepath)
        yield iteration, paths


def load_split_dataset(
    name: str,
    filename: DataTest | DataTrain,
    iteration: int,
) -> DataSet[DataDF] | DataSet[DataDF]:
    with open(
        sampling_filepath(
            name=name, filename=f"{iteration}_{filename}.pickle"
        ),
        mode="rb",
    ) as pickle_file:
        return cast(pd.DataFrame, pickle.load(pickle_file))
