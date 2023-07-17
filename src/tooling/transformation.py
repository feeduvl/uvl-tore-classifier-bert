from functools import partial
from typing import cast
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import omegaconf
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from strictly_typed_pandas import DataSet

from tooling.config import Transformation
from tooling.model import DataDF
from tooling.model import Label_None_Pad
from tooling.model import LABELS_NONE
from tooling.model import NoneLabel
from tooling.model import Token
from tooling.model import TORE_LABELS
from tooling.model import TORE_LEVEL
from tooling.model import ToreLabel
from tooling.model import ToreLevel


def get_class_weights(
    data: DataSet[DataDF],
) -> torch.Tensor:
    labels = np.array(data["tore_label"])
    unique_labels = np.unique(labels).tolist()
    unique_labels.sort(key=lambda x: LABELS_NONE.index(x))
    np_unique_labels = np.array(unique_labels)

    return torch.from_numpy(
        compute_class_weight(
            class_weight="balanced", classes=np_unique_labels, y=labels
        )
    ).to(torch.float32)


def lower_case_token(data: DataSet[DataDF]):
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


def transform_dataset(
    dataset: DataSet[DataDF], cfg: Transformation
) -> Tuple[List[Union[ToreLevel, ToreLabel, Literal["0"]]], DataSet[DataDF]]:
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

    return labels, dataset
