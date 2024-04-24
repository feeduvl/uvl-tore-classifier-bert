import pickle
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import cast
from typing import List
from typing import Literal
from typing import Tuple
from collections import Counter
from imblearn.over_sampling import SMOTEN

import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import GroupKFold
from strictly_typed_pandas import DataSet

from data import create_file
from data import sampling_filepath
from tooling.config import Experiment
from tooling.logging import logging_setup
from tooling.model import create_datadf, create_multi_datadf, get_id2label, get_label2id
from tooling.model import DataDF
from tooling.observability import log_artifacts

logging = logging_setup(__name__)

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
    if cfg_experiment.iterations == 1:
        data_train, data_test= create_multi_datadf(data=dataset)
        partial_files = (data_train, data_test)
        print(data_train)

        paths: List[Path] = []

        for partial_file, base_filename in zip(partial_files, FILES):
            filename = f"{1}_{base_filename}"

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

        #log_artifacts(paths)
        logging.info(
            f"Created datasets for foldless run"
        )

        yield 1, paths

    else:
        data = create_datadf(data=dataset)
        sentence_ids = dataset["sentence_id"]
        splitter = GroupKFold(
            n_splits=cfg_experiment.folds,
        )

        for iteration, (train_index, test_index) in enumerate(
            splitter.split(X=data, groups=sentence_ids)
        ):
            data_train = data.loc[train_index]
            
            if (cfg_experiment.smote):
                mlflow.log_param(f"labelDistribution_{iteration}_Original", Counter(data_train['tore_label']))
                data_train = apply_smote(data_train)
                mlflow.log_param(f"labelDistribution_{iteration}_SMOTE", Counter(data_train['tore_label']))
            
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

def apply_smote(traindatadf: pd.DataFrame) -> pd.DataFrame:
    """
    This function applies SMOTE to the pd.DataFrame "traindatadf" for balancing the dataset regarding 
    the number of label occurrences.

    Args:
        pd.DataFrame: The imbalanced panda dataframe.
    
    Returns:
        pd.DataFrame: The balanced panda dataframe after applying SMOTE.
    """

    id2label = get_id2label(traindatadf['tore_label'])
    label2id = get_label2id(traindatadf['tore_label'])

    dfNoZeroLabels = traindatadf[traindatadf['tore_label'] != "0"]

    strings = dfNoZeroLabels["string"].to_numpy().reshape(-1, 1)
    labels = dfNoZeroLabels["tore_label"].to_numpy()

    originalLength = len(strings)

    labelIDs = np.array([label2id[label] for label in labels])

    smote = SMOTEN()
    stringsSmoteList, labelIDsSmoteList = smote.fit_resample(strings, labelIDs)

    labelsSmoteList = np.array([id2label[id] for id in labelIDsSmoteList])

    smoteGeneratedStrings = stringsSmoteList[originalLength:]
    smoteGeneratedLabels = labelsSmoteList[originalLength:]

    lengthDiff = len(stringsSmoteList) - originalLength

    random_uuids = np.array([uuid.uuid4() for _ in range(lengthDiff)], dtype=object)
    sentenceIDs = np.array(random_uuids)
    sentenceIDXs = np.zeros(lengthDiff, dtype=int)

    smoteDfData = {
    'sentence_id': sentenceIDs,
    'sentence_idx': sentenceIDXs,
    'string': smoteGeneratedStrings.flatten(),
    'tore_label': smoteGeneratedLabels
    }

    trainDataDfSMOTE = pd.DataFrame(smoteDfData)

    balancedTrainDataDf = pd.concat([traindatadf, trainDataDfSMOTE], ignore_index=True)

    return balancedTrainDataDf
