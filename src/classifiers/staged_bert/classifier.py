import itertools
from functools import partial
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import TypedDict

import mlflow
import numpy as np
import numpy.typing as npt
import omegaconf
from datasets import Dataset
from strictly_typed_pandas.dataset import DataSet
from transformers import BertForTokenClassification
from transformers import BertTokenizerFast
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction

from classifiers.bert.classifier import create_bert_dataset
from classifiers.bert.classifier import Modification
from classifiers.bert.classifier import setup_device
from classifiers.staged_bert.model import StagedBertForTokenClassification
from tooling.config import Transformation
from tooling.logging import logging_setup
from tooling.model import DataDF
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


def classify_with_bert_stage_1(
    model_path: Path,
    data: DataSet[DataDF],
    label2id: Dict[Label_None_Pad, int],
    tokenizer: BertTokenizerFast,
    max_len: int,
) -> Dataset:
    logging.info("Converting dataframe to bert dataset")
    bert_data = create_bert_dataset(
        input_data=data,
        label2id=label2id,
        tokenizer=tokenizer,
        max_len=max_len,
    )

    if not bert_data:
        raise ValueError("No BERT Dataset supplied")

    logging.info("Loading Model")
    model = BertForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=model_path
    )
    model.to(device=setup_device())
    trainer = Trainer(model=model)

    logging.info("Creating hint column")
    # Add predicted hint labels to train_dataset
    first_stage_train_result = trainer.predict(
        bert_data.remove_columns("labels")
    )
    hint_column = get_hint_column(first_stage_train_result[0])
    bert_data = bert_data.add_column("hint_input_ids", hint_column)
    return bert_data


def hint_column_to_labels(
    column: List[List[np.int64]], id2label: Dict[int, Label_None_Pad]
) -> List[List[Label_None_Pad]]:
    labels: List[List[Label_None_Pad]] = []

    for sentence in column:
        sentence_labels: List[Label_None_Pad] = []
        for token in sentence:
            sentence_labels.append(id2label[token])
        labels.append(sentence_labels)

    return labels


def classify_with_bert_stage_2(
    model_path: Path, bert_data: Dataset, id2label: Dict[int, Label_None_Pad]
) -> List[List[Label_None_Pad]]:
    logging.info("Loading Model")
    model = StagedBertForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=model_path
    )
    model.to(device=setup_device())
    trainer = Trainer(model=model)

    logging.info("Creating results")
    # Add predicted hint labels to train_dataset
    first_stage_train_result = trainer.predict(
        bert_data.remove_columns("labels")
    )
    hint_column = get_hint_column(first_stage_train_result[0])

    return hint_column_to_labels(column=hint_column, id2label=id2label)


def classify_with_bert(
    model_path: Path,
    bert_data: Dataset,
    id2label: Dict[int, Label_None_Pad],
) -> List[List[Label_None_Pad]]:
    logging.info("Loading Model")
    model = BertForTokenClassification.from_pretrained(
        pretrained_model_name_or_path=model_path
    )
    model.to(device=setup_device())
    trainer = Trainer(model=model)

    logging.info("Creating hint column")
    # Add predicted hint labels to train_dataset
    first_stage_train_result = trainer.predict(
        bert_data.remove_columns("labels")
    )
    hint_column = get_hint_column(first_stage_train_result[0])

    return hint_column_to_labels(column=hint_column, id2label=id2label)
