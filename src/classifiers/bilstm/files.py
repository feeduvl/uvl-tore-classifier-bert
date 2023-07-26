import pickle
from pathlib import Path
from typing import cast

from strictly_typed_pandas import DataSet

from data import bilstm_filepath
from tooling.model import ResultDF


MODEL_FILENAME = "model"


def model_path(name: str, iteration: int) -> Path:
    return bilstm_filepath(
        name=name,
        filename=f"{iteration}_" + MODEL_FILENAME,
    )
