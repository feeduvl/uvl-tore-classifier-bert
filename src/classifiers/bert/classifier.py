import itertools
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from typing import cast
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TypedDict

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from strictly_typed_pandas import DataSet
from torch import nn
from transformers import BatchEncoding
from transformers import BertTokenizerFast
from transformers import Trainer
from transformers.trainer_utils import EvalPrediction

from tooling import evaluation
from tooling.config import BERT
from tooling.evaluation import IterationResult
from tooling.logging import logging_setup
from tooling.model import create_resultdf
from tooling.model import data_to_list_of_label_lists
from tooling.model import data_to_list_of_token_lists
from tooling.model import DataDF
from tooling.model import Label_None_Pad
from tooling.model import LABELS_NONE
from tooling.model import ResultDF

logging = logging_setup()


class BertData(TypedDict):
    id: int
    string: List[str]
    tore_label_id: List[int]


class BertDatas(TypedDict):
    id: List[int]
    string: List[List[str]]
    tore_label_id: List[List[int]]


class StagedBertDatas(TypedDict):
    id: List[int]
    string: List[List[str]]
    tore_label_id: List[List[int]]
    hint_label_id: List[List[int]]


def prepare_data(
    dataframe: DataSet[DataDF], label2id: Dict[Label_None_Pad, int]
) -> List[BertData]:
    token_lists_list = data_to_list_of_token_lists(dataframe)
    label_lists_list = data_to_list_of_label_lists(
        data=dataframe, label2id=label2id
    )

    data: List[BertData] = [
        {"id": id, "string": data[0], "tore_label_id": data[1]}
        for id, data in enumerate(zip(token_lists_list, label_lists_list))
    ]

    return data


def tokenize_and_align_labels(
    data: BertDatas | StagedBertDatas,
    tokenizer: BertTokenizerFast,
    max_len: Optional[int],
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

    for column in columns:
        labels = []

        for i, label in enumerate(data[column]):  # type:ignore
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs[column] = labels

    return tokenized_inputs


def get_tokenize_and_align_labels(
    tokenizer: BertTokenizerFast,
    max_len: Optional[int],
    truncation: bool = True,
) -> partial[BatchEncoding]:
    func = partial(
        tokenize_and_align_labels,
        tokenizer=tokenizer,
        max_len=max_len,
        truncation=truncation,
    )
    return func


def compute_prediction_and_solution(
    p: EvalPrediction, id2label: Dict[int, str]
) -> Tuple[DataSet[ResultDF], DataSet[ResultDF]]:
    predictions, labels = p
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
    run_name: str,
    id2label: Dict[int, str],
) -> Dict[str, float]:
    prediction, solution = compute_prediction_and_solution(
        p, id2label=id2label
    )

    iteration = len(iteration_tracking)

    iteration_result = evaluation.evaluate_iteration(
        run_name=run_name,
        iteration=iteration,
        average=average,
        solution=solution,
        result=prediction,
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
) -> partial[Dict[str, float]]:
    return partial(
        compute_metrics,
        iteration_tracking=iteration_tracking,
        average=average,
        run_name=run_name,
        id2label=id2label,
    )


def create_tore_dataset(
    data: DataSet[DataDF], label2id: Dict[Label_None_Pad, int]
) -> Dataset:
    prepared_data = prepare_data(data, label2id=label2id)
    prepared_data_df = pd.DataFrame.from_records(prepared_data)
    return Dataset.from_pandas(df=prepared_data_df)


class WeightedTrainer(Trainer):  # type: ignore
    def __init__(self, *args, **kwargs):  # type: ignore
        super().__init__(*args, **kwargs)
        self.class_weights = kwargs["class_weights"]

    def compute_loss(self, model, inputs, return_outputs=False):  # type: ignore
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)

        weights = self.class_weights
        weights.to("mps")

        loss_fct = nn.CrossEntropyLoss(weight=weights)

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def setup_device() -> str:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    return device


def get_class_weights(
    bert_cfg: BERT, data: DataSet[DataDF], device: str
) -> torch.Tensor:
    labels = np.array(data["tore_label"])
    unique_labels = np.unique(labels).tolist()
    unique_labels.sort(key=lambda x: LABELS_NONE.index(x))

    if bert_cfg.weighted_classes:
        np_unique_labels = np.array(unique_labels)
        class_weights = torch.from_numpy(
            compute_class_weight(
                class_weight="balanced", classes=np_unique_labels, y=labels
            )
        ).to(torch.float32)
    else:
        weights = [1.0 for label in unique_labels]
        class_weights = torch.from_numpy(np.array(weights)).to(torch.float32)

    logging.info(f"Class weights: {dict(zip(unique_labels, class_weights))}")

    class_weights.to(device=device)
    return class_weights


def get_max_len(
    bert_cfg: BERT,
    data: DataSet[DataDF],
    label2id: Dict[Label_None_Pad, int],
    tokenizer: BertTokenizerFast,
) -> int:
    if bert_cfg.max_len:
        max_len = bert_cfg.max_len
        logging.info(f"Configured maximal token sequence length: {max_len = }")
        mlflow.log_param("max_len", max_len)
    else:
        t_a_a_l_test = get_tokenize_and_align_labels(
            tokenizer=tokenizer, max_len=None, truncation=False
        )
        test_data = create_tore_dataset(data=data, label2id=label2id)
        test_data = test_data.map(
            t_a_a_l_test,
            batched=True,
        )
        test_data = test_data.rename_columns(
            {"string": "text", "tore_label_id": "labels"}
        )

        max_len = max([len(s) for s in test_data["text"]])
        logging.info(f"Computed maximal token sequence length: {max_len = }")
        mlflow.log_param("computed_max_len", max_len)

    return max_len
