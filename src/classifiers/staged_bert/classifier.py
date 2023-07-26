import itertools
from functools import partial
from typing import Dict
from typing import List
from typing import Optional
from typing import TypedDict

import mlflow
import numpy as np
import numpy.typing as npt
import omegaconf
from datasets import Dataset
from transformers.trainer_utils import EvalPrediction

from classifiers.bert.classifier import Modification
from tooling.config import Transformation
from tooling.logging import logging_setup
from tooling.model import get_label2id
from tooling.model import Label_None_Pad
from tooling.model import LABELS_NONE
from tooling.model import ZERO
from tooling.transformation import transform_token_label

logging = logging_setup(__name__)


def generate_hint_data(
    dataset: Dataset,
    column: str,
    id2label: Dict[int, Label_None_Pad],
    hint_transformation: partial[Optional[Label_None_Pad]],
    hint_label2id: Dict[Label_None_Pad, int],
) -> List[List[int]]:
    hints = []
    tore_label_ids = dataset[column]

    for tore_label_list in tore_label_ids:
        hint_list = []
        for label_id in tore_label_list:
            if label_id == -100:
                hint_list.append(-100)
                continue
            label = id2label[label_id]
            transformed_label = hint_transformation(label)
            if transformed_label is None:
                hint_list.append(hint_label2id[ZERO])
            else:
                hint_list.append(hint_label2id[transformed_label])

        assert len(hint_list) == len(tore_label_list)
        hints.append(hint_list)

    return hints


class Hints(TypedDict):
    transformation_function: partial[Optional[Label_None_Pad]]
    label2id: Dict[Label_None_Pad, int]


def get_hint_transformation(
    transformation_cfg: Transformation,
) -> Hints:
    dict_cfg = omegaconf.OmegaConf.to_container(transformation_cfg)

    if not isinstance(dict_cfg, dict):
        raise ValueError("No config passed")

    del dict_cfg["description"]
    del dict_cfg["type"]

    hint_labels: List[Label_None_Pad] = ["0"]

    for new_value in dict_cfg.values():
        if new_value is None:
            continue
        elif new_value in LABELS_NONE:
            hint_labels.append(new_value)
            continue
        else:
            raise ValueError(
                f"Transformation value '{new_value}' isn't valid TORE_LABEL"
            )

    hint_label2id = get_label2id(list(set(hint_labels)))
    transformation_function = partial(
        transform_token_label, cfg=transformation_cfg
    )

    logging.info(f"Hint Label2Id: {hint_label2id=}")
    mlflow.log_param("hint_label2id", hint_label2id)

    return Hints(
        transformation_function=transformation_function, label2id=hint_label2id
    )


def get_hint_modifier(
    id2label: Dict[int, Label_None_Pad], hints: Hints
) -> Modification:
    column_name = "hint_input_ids"
    func = partial(
        generate_hint_data,
        column="tore_label_id",
        id2label=id2label,
        hint_transformation=hints["transformation_function"],
        hint_label2id=hints["label2id"],
    )
    return Modification(column_name=column_name, modifier=func)


def get_hint_column(
    predictions: npt.NDArray[np.float64],
) -> List[List[np.int64]]:
    predictions = np.argmax(predictions, axis=2)

    predictions_list = [[p for p in prediction] for prediction in predictions]

    return predictions_list
