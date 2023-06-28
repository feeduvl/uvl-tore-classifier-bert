from typing import Optional

import omegaconf
import pandas as pd
from strictly_typed_pandas import DataSet
from tooling.model import DataDF
from tooling.model import Token
from tooling.model import TORE_LABELS
from tooling.model import ToreLabel


def transform_token_label(
    token_label: ToreLabel | None, cfg: dict[str, str]
) -> Optional[ToreLabel]:
    if token_label is None:
        return None

    new_value = cfg.get(token_label.lower(), None)

    if new_value is None:
        return None

    if new_value == token_label.lower():
        return token_label

    return new_value


def transform_dataset(
    dataset: DataSet[DataDF], cfg: omegaconf.DictConfig
) -> DataSet[DataDF]:
    dict_cfg = omegaconf.OmegaConf.to_container(cfg)

    del dict_cfg["description"]

    for new_value in dict_cfg.values():
        if new_value in TORE_LABELS:
            continue
        elif new_value is None:
            continue
        else:
            raise ValueError(
                f"Transformation value '{new_value}' isn't valid TORE_LABEL"
            )

    dataset["tore_label"] = dataset["tore_label"].apply(
        transform_token_label, cfg=cfg
    )

    return dataset
