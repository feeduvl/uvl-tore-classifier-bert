from typing import cast
from typing import List
from typing import Literal
from typing import Optional
from typing import TypedDict
from typing import Union, Dict
from functools import partial
from tooling.model import get_label2id
import mlflow
import numpy as np
import numpy.typing as npt
import omegaconf
from sklearn.utils.class_weight import compute_class_weight
from strictly_typed_pandas import DataSet

from tooling.config import BERT
from tooling.config import BiLSTM
from tooling.config import Config
from tooling.config import Transformation
from tooling.loading import load_dataset
from tooling.logging import logging_setup
from tooling.model import DataDF
from tooling.model import Label_None
from tooling.model import Label_None_Pad
from tooling.model import LABELS_NONE
from tooling.model import NoneLabel
from tooling.model import TORE_LABELS
from tooling.model import TORE_LEVEL
from tooling.model import ToreLabel
from tooling.model import ToreLevel
from tooling.model import ZERO

from tooling.observability import log_param

logging = logging_setup(__name__)


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
    log_param("dataset_labels", transformed_dataset["labels"])

    return transformed_dataset


def get_class_weights(
    exp_cfg: BERT | BiLSTM,
    data: DataSet[DataDF],
) -> npt.NDArray[np.float32]:
    labels = np.array(data["tore_label"])
    unique_labels = np.unique(labels).tolist()
    unique_labels.sort(key=lambda x: LABELS_NONE.index(x))

    class_weights: npt.NDArray[np.float32]

    if exp_cfg.weighted_classes:
        np_unique_labels = np.array(unique_labels)
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np_unique_labels, y=labels
        ).astype(np.float32)

    else:
        weights = [1.0 for label in unique_labels]
        class_weights = np.array(weights).astype(np.float32)

    logging.info(f"Class weights: {dict(zip(unique_labels, class_weights))}")

    return class_weights


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

    return Hints(
        transformation_function=transformation_function, label2id=hint_label2id
    )
