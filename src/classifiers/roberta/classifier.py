import itertools
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypedDict

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from datasets import Dataset
from strictly_typed_pandas import DataSet
from torch import nn
from transformers import BatchEncoding
from transformers import RobertaTokenizerFast
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import EvalPrediction

from tooling import evaluation
from tooling.config import RoBERTa
from tooling.logging import logging_setup
from tooling.model import create_resultdf
from tooling.model import data_to_list_of_label_lists
from tooling.model import data_to_list_of_token_lists
from tooling.model import DataDF
from tooling.model import HintedDataDF
from tooling.model import Label_None
from tooling.model import Label_None_Pad
from tooling.model import ResultDF
from tooling.types import IterationResult

logging = logging_setup(__name__)


class RoBertaData(TypedDict):
    id: int
    string: List[str]
    tore_label_id: List[int]


class StagedRoBertaData(TypedDict):
    id: int
    string: List[str]
    tore_label_id: List[int]
    hint_label_id: List[int]


class RoBertaDatas(TypedDict):
    id: List[int]
    string: List[List[str]]
    tore_label_id: List[List[int]]


class StagedRoBertaDatas(TypedDict):
    id: List[int]
    string: List[List[str]]
    tore_label_id: List[List[int]]
    hint_label_id: List[List[int]]


def prepare_data(
    dataframe: DataSet[DataDF],
    label2id: Dict[Label_None_Pad, int],
    ignore_labels: bool,
) -> List[RoBertaData]:
    token_lists_list = data_to_list_of_token_lists(dataframe)
    if not ignore_labels:
        label_lists_list = data_to_list_of_label_lists(
            data=dataframe, label2id=label2id
        )

        data_with_labels: List[RoBertaData] = [
            {"id": id, "string": data[0], "tore_label_id": data[1]}
            for id, data in enumerate(zip(token_lists_list, label_lists_list))
        ]
        return data_with_labels

    else:
        data_without_labels: List[RoBertaData] = [
            {"id": id, "string": data[0], "tore_label_id": None}  # type: ignore
            for id, data in enumerate(zip(token_lists_list))
        ]

        return data_without_labels


def pad_or_truncate_list(input_list: List[int], length: int) -> List[int]:
    if len(input_list) < length:
        diff = length - len(input_list)
        output_list = input_list + ([0] * diff)
    elif len(input_list) == length:
        return input_list
    elif len(input_list) > length:
        return input_list[:length]

    return output_list


def prepare_data_with_hints(
    dataframe: DataSet[DataDF] | DataSet[HintedDataDF],
    label2id: Dict[Label_None_Pad, int],
    hint_label2id: Dict[Label_None_Pad, int],
    ignore_labels: bool,
) -> List[StagedRoBertaData]:
    token_lists_list = data_to_list_of_token_lists(dataframe)
    hint_id_lists_list = data_to_list_of_label_lists(
        data=dataframe, label2id=hint_label2id, column="hint_input_ids"
    )

    if not ignore_labels:
        label_lists_list = data_to_list_of_label_lists(
            data=dataframe, label2id=label2id
        )
    else:
        label_lists_list = [None] * len(token_lists_list)  # type: ignore

    data: List[StagedRoBertaData] = [
        {
            "id": id,
            "string": data[0],
            "tore_label_id": data[1],
            "hint_label_id": data[2],
        }
        for id, data in enumerate(
            zip(
                token_lists_list,
                label_lists_list,
                hint_id_lists_list,
                strict=True,
            )
        )
    ]

    return data


def tokenize_and_align_labels(
    data: RoBertaDatas | StagedRoBertaDatas,
    tokenizer: RobertaTokenizerFast,
    max_len: Optional[int],
    align_labels: bool,
    truncation: bool = True,
) -> BatchEncoding:
    tokenized_inputs = tokenizer(
        data["string"],
        truncation=truncation,
        max_length=max_len,
        is_split_into_words=True,
        padding="max_length",
    )

    columns = list(data.keys())
    columns.remove("id")
    columns.remove("string")

    if not align_labels:
        columns.remove("tore_label_id")

    try:
        padding_value = -100
        for column in columns:
            labels = []
            if column == "hint_label_id":
                padding_value = 0

            for i, label in enumerate(data[column]):  # type:ignore
                word_ids = tokenized_inputs.word_ids(
                    batch_index=i
                )  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(padding_value)
                    elif (
                        word_idx != previous_word_idx
                    ):  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(padding_value)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs[column] = labels
    except TypeError as e:
        if align_labels:
            raise e
        else:
            pass

    return tokenized_inputs


def get_tokenize_and_align_labels(
    tokenizer: RobertaTokenizerFast,
    max_len: Optional[int],
    truncation: bool = True,
    align_labels: bool = True,
) -> partial[BatchEncoding]:
    func = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        max_len=max_len,
        truncation=truncation,
        align_labels=align_labels,
    )
    return func


def compute_prediction_and_solution(
    predictions: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    id2label: Dict[int, Label_None_Pad],
) -> Tuple[DataSet[ResultDF], DataSet[ResultDF]]:
    predictions = np.argmax(predictions, axis=2)

    predictions_list = list(
        itertools.chain.from_iterable(
            [
                [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
        )
    )

    df_predictions = pd.DataFrame(predictions_list, columns=["tore_label"])
    df_predictions["string"] = ""

    solutions_list = list(
        itertools.chain.from_iterable(
            [
                [
                    id2label[cast(int, l)]
                    for (p, l) in zip(prediction, label)
                    if l != -100
                ]
                for prediction, label in zip(predictions, labels)
            ]
        )
    )

    df_solutions = pd.DataFrame(solutions_list, columns=["tore_label"])
    df_solutions["string"] = ""

    return create_resultdf(df_predictions), create_resultdf(df_solutions)


def compute_metrics(
    p: EvalPrediction,
    iteration_tracking: List[IterationResult],
    average: str,
    labels: List[Label_None],
    run_name: str,
    id2label: Dict[int, Label_None_Pad],
    create_confusion_matrix: bool,
) -> Dict[str, float]:
    prediction, solution = compute_prediction_and_solution(
        p[0], p[1], id2label=id2label
    )

    iteration = len(iteration_tracking)

    iteration_result = evaluation.evaluate_iteration(
        iteration_tracking=iteration_tracking,
        run_name=run_name,
        iteration=iteration,
        average=average,
        labels=labels,
        solution=solution,
        result=prediction,
        create_confusion_matrix=create_confusion_matrix,
    )

    iteration_tracking.append(iteration_result)

    report_result = deepcopy(asdict(iteration_result))

    del report_result["confusion_matrix"]
    del report_result["result"]
    del report_result["solution"]
    del report_result["pl_precision"]
    del report_result["pl_recall"]

    return report_result


def get_compute_metrics(
    iteration_tracking: List[IterationResult],
    average: str,
    run_name: str,
    id2label: Dict[int, Label_None_Pad],
    labels: List[Label_None],
    create_confusion_matrix: bool,
) -> partial[Dict[str, float]]:
    return partial(
        compute_metrics,
        iteration_tracking=iteration_tracking,
        average=average,
        run_name=run_name,
        id2label=id2label,
        labels=labels,
        create_confusion_matrix=create_confusion_matrix,
    )


class Modification(TypedDict):
    column_name: str
    modifier: partial[List[List[int]]]


def create_roberta_dataset(
    input_data: DataSet[DataDF],
    label2id: Dict[Label_None_Pad, int],
    tokenizer: RobertaTokenizerFast,
    max_len: int,
    modifiers: List[Modification] = [],
    ignore_labels: bool = False,
    align_labels: bool = True,
) -> Dataset:
    prepared_data = prepare_data(
        input_data,
        label2id=label2id,
        ignore_labels=ignore_labels,
    )
    prepared_data_df = pd.DataFrame.from_records(prepared_data)
    data = Dataset.from_pandas(df=prepared_data_df)

    if modifiers:
        for modifier in modifiers:
            data = data.add_column(
                modifier["column_name"], modifier["modifier"](dataset=data)
            )

    tokenizer_and_aligner = get_tokenize_and_align_labels(
        tokenizer=tokenizer,
        max_len=max_len,
        truncation=True,
        align_labels=align_labels,
    )
    data = data.map(tokenizer_and_aligner, batched=True)

    data = data.rename_columns({"string": "text", "tore_label_id": "labels"})

    return data


def create_staged_roberta_dataset(
    input_data: DataSet[HintedDataDF],
    label2id: Dict[Label_None_Pad, int],
    hint_label2id: Dict[Label_None_Pad, int],
    tokenizer: RobertaTokenizerFast,
    max_len: int,
    ignore_labels: bool = False,
    align_labels: bool = True,
) -> Dataset:
    prepared_data = prepare_data_with_hints(
        input_data,
        label2id=label2id,
        hint_label2id=hint_label2id,
        ignore_labels=ignore_labels,
    )
    prepared_data_df = pd.DataFrame.from_records(prepared_data)
    data = Dataset.from_pandas(df=prepared_data_df)

    tokenizer_and_aligner = get_tokenize_and_align_labels(
        tokenizer=tokenizer,
        max_len=max_len,
        truncation=True,
        align_labels=align_labels,
    )
    data = data.map(tokenizer_and_aligner, batched=True)

    data = data.rename_columns(
        {
            "string": "text",
            "tore_label_id": "labels",
            "hint_label_id": "hint_input_ids",
        }
    )

    return data


# mypy: allow-untyped-defs
class WeightedTrainer(Trainer):
    # mypy: allow-untyped-defs
    def __init__(
        self,
        class_weights,
        device,
        learning_rate_roberta,
        learning_rate_classifier,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(device=device)
        self.learning_rate_roberta = learning_rate_roberta
        self.learning_rate_classifier = learning_rate_classifier

    # mypy: allow-untyped-defs
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)

        weights = self.class_weights
        weights.to(setup_device())

        loss_fct = nn.CrossEntropyLoss(weight=weights)

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        decay_parameters = get_parameter_names(
            self.model, ALL_LAYERNORM_LAYERS
        )
        decay_parameters = [
            name for name in decay_parameters if "bias" not in name
        ]

        pretrained = self.model.roberta.parameters()
        pretrained_names = [
            f"roberta.{k}" for (k, v) in self.model.roberta.named_parameters()
        ]

        new_params = [
            v
            for k, v in self.model.named_parameters()
            if k not in pretrained_names
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )

        optimizer_kwargs["params"] = [
            {"params": pretrained},
            {"params": new_params, "lr": self.learning_rate_classifier},
        ]

        optimizer_kwargs["lr"] = self.learning_rate_roberta

        self.optimizer = optimizer_cls(**optimizer_kwargs)

        return self.optimizer


def setup_device() -> str:
    device = "cpu"
    device = "cuda" if torch.cuda.is_available() else device
    device = "mps" if torch.backends.mps.is_available() else device

    logging.debug(f"Using device: {device}")
    return device


def get_max_len(
    roberta_cfg: RoBERTa,
    data: DataSet[DataDF],
    label2id: Dict[Label_None_Pad, int],
    tokenizer: RobertaTokenizerFast,
) -> int:
    if roberta_cfg.max_len:
        max_len = roberta_cfg.max_len
        logging.info(f"Configured maximal token sequence length: {max_len = }")
        mlflow.log_param("max_len", max_len)
    else:
        t_a_a_l_test = get_tokenize_and_align_labels(
            tokenizer=tokenizer, max_len=None, truncation=False
        )
        prepared_data = prepare_data(
            data, label2id=label2id, ignore_labels=False
        )
        prepared_data_df = pd.DataFrame.from_records(prepared_data)
        test_data = Dataset.from_pandas(df=prepared_data_df)
        test_data = test_data.map(
            t_a_a_l_test,
            batched=True,
        )
        test_data = test_data.rename_columns(
            {"string": "text", "tore_label_id": "labels"}
        )

        # iterate over all input_ids and remove the tokens with the special embedding value 0 as these are used for padding
        # get the longest sequence and add 1
        # that is the longest sequence the classifier has to handle
        max_len = (
            max(
                [
                    len(list(filter((0).__ne__, s)))
                    for s in test_data["input_ids"]
                ]
            )
            + 1
        )
        logging.info(f"Computed maximal token sequence length: {max_len = }")
        mlflow.log_param("computed_max_len", max_len)

    return max_len
