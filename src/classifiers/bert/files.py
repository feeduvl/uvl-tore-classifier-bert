import pickle
import shutil
from pathlib import Path
from typing import cast

from strictly_typed_pandas import DataSet

from data import bert_filepath
from tooling.model import ResultDF

MODEL_FILENAME = "bert"
OUTPUT_FILENAME = "output"
LOGGING_FILENAME = "logging"


def model_path(name: str, iteration: int) -> Path:
    return bert_filepath(
        name=name,
        filename=f"{iteration}_" + MODEL_FILENAME,
    )


def output_path(name: str, clean: bool) -> Path:
    return bert_filepath(name=name, filename=OUTPUT_FILENAME, clean=clean)


def logging_path(name: str, clean: bool) -> Path:
    return bert_filepath(name=name, filename=LOGGING_FILENAME, clean=clean)
