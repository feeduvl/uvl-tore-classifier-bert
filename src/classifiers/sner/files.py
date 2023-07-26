import pickle
from pathlib import Path
from typing import cast

from strictly_typed_pandas import DataSet

from data import sner_filepath
from data import SNER_TEMP
from tooling.model import ResultDF

BASE_PATH = Path(__file__).parent
RESSOURCES_PATH = BASE_PATH.joinpath(Path("./ressources"))
STANFORD_JAR_PATH = RESSOURCES_PATH.joinpath("./stanford-ner.jar")


TEMPLATE_FILENAME = "ner_training.prop.j2"

TRAIN_FILENAME = "sner_train_file"
CONFIG_FILENAME = "sner_config_file"
MODEL_FILENAME = "model"
RESULT_FILENAME = "classification_result"
SOLUTION_FILENAME = "solution"


def trainfile(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name,
        filename=f"{iteration}_" + TRAIN_FILENAME + ".txt",
    )


def configfile(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name,
        filename=f"{iteration}_" + CONFIG_FILENAME + ".prop",
    )


def modelfile(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name,
        filename=f"{iteration}_" + MODEL_FILENAME + ".ser.gz",
    )


def resultfile_csv(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name,
        filename=f"{iteration}_" + RESULT_FILENAME + ".csv",
    )


def resultfile_pickle(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name, filename=f"{iteration}_" + RESULT_FILENAME + ".pickle"
    )


def solutionfile_pickle(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name,
        filename=f"{iteration}_" + SOLUTION_FILENAME + ".pickle",
    )


def solutionfile_csv(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name,
        filename=f"{iteration}_" + SOLUTION_FILENAME + ".csv",
    )


def load_result(name: str, iteration: int) -> DataSet[ResultDF]:
    with open(
        resultfile_pickle(name=name, iteration=iteration), mode="rb"
    ) as pickle_file:
        dataset = pickle.load(pickle_file)
    return cast(DataSet[ResultDF], dataset)


def load_solution(name: str, iteration: int) -> DataSet[ResultDF]:
    with open(
        solutionfile_pickle(name=name, iteration=iteration), mode="rb"
    ) as pickle_file:
        dataset = pickle.load(pickle_file)

    return cast(DataSet[ResultDF], dataset)
