from functools import partial
from typing import cast
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import TypedDict
from typing import Union

import mlflow
import numpy as np
import omegaconf
import pandas as pd
import torch
from strictly_typed_pandas import DataSet

from tooling.config import Config
from tooling.config import Transformation
from tooling.loading import load_dataset
from tooling.model import DataDF
from tooling.model import Label_None
from tooling.model import Label_None_Pad
from tooling.model import LABELS_NONE
from tooling.model import NoneLabel
from tooling.model import Token
from tooling.model import TORE_LABELS
from tooling.model import TORE_LEVEL
from tooling.model import ToreLabel
from tooling.model import ToreLevel
from tooling.model import ZERO

logging = logging_setup()


def lower_case_token(data: DataSet[DataDF]) -> None:
    data["string"] = data["string"].apply(str.lower)


def transform_token_label(
    token_label: Label_None_Pad, cfg: dict[str, str]
) -> Optional[Label_None_Pad]:
    if token_label is None:
        return None

    new_value = cast(
        Union[ToreLevel, ToreLabel, None, NoneLabel],
        cfg.get(token_label.lower(), None),
    )

    if new_value is None:
        return None

    if new_value == token_label.lower():
        return token_label

    return new_value


class TransformedDataset(TypedDict):
    labels: List[Label_None]
    dataset: DataSet[DataDF]


def _transform_dataset(
    dataset: DataSet[DataDF], cfg: Transformation
) -> TransformedDataset:
    dict_cfg = omegaconf.OmegaConf.to_container(cfg)

    if not isinstance(dict_cfg, dict):
        raise ValueError("No config passed")

    del dict_cfg["description"]
    del dict_cfg["type"]

    labels: List[Union[ToreLevel, ToreLabel, Literal["0"]]] = ["0"]

    for new_value in dict_cfg.values():
        if new_value is None:
            continue
        elif new_value in TORE_LABELS:
            labels.append(new_value)
            continue
        elif new_value in TORE_LEVEL:
            labels.append(new_value)
            continue
        else:
            raise ValueError(
                f"Transformation value '{new_value}' isn't valid TORE_LABEL"
            )

    dataset["tore_label"] = dataset["tore_label"].apply(
        transform_token_label, cfg=cfg
    )

    labels = list(set(labels))

    return TransformedDataset(labels=labels, dataset=dataset)


def transform_dataset(
    cfg: Config, run_name: str, fill_with_zeros: bool
) -> TransformedDataset:
    d = load_dataset(name=run_name)

    transformed_dataset = _transform_dataset(dataset=d, cfg=cfg.transformation)

    if fill_with_zeros:
        transformed_dataset["dataset"].fillna(ZERO, inplace=True)

    if cfg.experiment.lower_case:
        lower_case_token(transformed_dataset["dataset"])

    logging.info(f"Dataset Labels: {transformed_dataset['labels']=}")
    mlflow.log_param("dataset_labels", transformed_dataset["labels"])

    return transformed_dataset
