import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from typing import List
from typing import Literal
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupKFold
from strictly_typed_pandas import DataSet

from data import create_file
from data import sampling_filepath
from tooling.config import Experiment
from tooling.logging import logging_setup
from tooling.model import create_datadf
from tooling.model import DataDF
from tooling.observability import log_artifacts

logging = logging_setup()

DataTrain = Literal["data_train"]
DATA_TRAIN: DataTrain = "data_train"

DataTest = Literal["data_test"]
DATA_TEST: DataTest = "data_test"


FILES = (
    DATA_TRAIN,
    DATA_TEST,
)


def split_dataset_k_fold(
    name: str, dataset: pd.DataFrame, cfg_experiment: Experiment
) -> Iterator[Tuple[int, List[Path]]]:
    data = create_datadf(data=dataset)
    sentence_ids = dataset["sentence_id"]

    splitter = GroupKFold(
        n_splits=cfg_experiment.folds,
    )
    for iteration, (train_index, test_index) in enumerate(
        splitter.split(X=data, groups=sentence_ids)
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

        log_artifacts(paths)
        logging.info(
            f"Created fold datasets for fold: {iteration}, stored at {paths=}"
        )

        yield iteration, paths


def load_split_dataset(
    name: str,
    filename: DataTest | DataTrain,
    iteration: int,
) -> DataSet[DataDF]:
    with open(
        sampling_filepath(
            name=name, filename=f"{iteration}_{filename}.pickle"
        ),
        mode="rb",
    ) as pickle_file:
        return cast(DataSet[DataDF], pickle.load(pickle_file))
