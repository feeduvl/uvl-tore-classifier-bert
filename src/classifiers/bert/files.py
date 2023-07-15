import pickle
from pathlib import Path
from typing import cast

from strictly_typed_pandas import DataSet

from data import bert_filepath
from tooling.model import ResultDF


MODEL_FILENAME = "bert"
OUTPUT_FILENAME = "output"


def model_path(name: str, iteration: int) -> Path:
    return bert_filepath(
        name=name,
        filename=f"{iteration}_" + MODEL_FILENAME,
    )


def output_path(name: str, iteration: int) -> Path:
    return bert_filepath(
        name=name,
        filename=f"{iteration}_" + OUTPUT_FILENAME,
    )
