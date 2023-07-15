import itertools
from copy import deepcopy
from dataclasses import asdict
from functools import partial
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Tuple
from typing import TypedDict

import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from strictly_typed_pandas import DataSet
from torch.optim import Optimizer
from transformers import AutoTokenizer
from transformers import BatchEncoding
from transformers import BertForTokenClassification
from transformers import BertTokenizerFast
from transformers.trainer_utils import EvalPrediction

from tooling import evaluation
from tooling.evaluation import IterationResult
from tooling.model import create_resultdf
from tooling.model import data_to_list_of_label_lists
from tooling.model import data_to_list_of_token_lists
from tooling.model import DataDF
from tooling.model import id_to_label
from tooling.model import Label_None_Pad
from tooling.model import label_to_id
from tooling.model import ResultDF
from tooling.model import TokenizedDataDF
from tooling.model import ZERO
from tooling.observability import log_iteration_result


class BertData(TypedDict):
    id: int
    string: List[str]
    tore_label_id: List[int]


class BertDatas(TypedDict):
    id: List[int]
    string: List[List[str]]
    tore_label_id: List[List[int]]


def prepare_data(dataframe: DataSet[DataDF]) -> List[BertData]:
    token_lists_list = data_to_list_of_token_lists(dataframe)
    label_lists_list = data_to_list_of_label_lists(
        data=dataframe, use_label_ids=True
    )

    data = [
        BertData(id=id, string=data[0], tore_label_id=data[1])
        for id, data in enumerate(zip(token_lists_list, label_lists_list))
    ]

    return data


def tokenize_and_align_labels(
    data: BertDatas, tokenizer: BertTokenizerFast, max_len: int
) -> BatchEncoding:
    tokenized_inputs = tokenizer(
        data["string"],
        truncation=True,
        is_split_into_words=True,
    )
    labels = []

    for i, label in enumerate(data["tore_label_id"]):
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

    tokenized_inputs["tore_label_id"] = labels
    return tokenized_inputs


def get_tokenize_and_align_labels(
    tokenizer: BertTokenizerFast, max_len: int
) -> partial[BatchEncoding]:
    func = partial(
        tokenize_and_align_labels, tokenizer=tokenizer, max_len=max_len
    )
    return func


def compute_prediction_and_solution(
    p: EvalPrediction,
) -> Tuple[DataSet[ResultDF], DataSet[ResultDF]]:
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    predictions_list = list(
        itertools.chain.from_iterable(
            [
                [
                    id_to_label(p)
                    for (p, l) in zip(prediction, label)
                    if l != -100
                ]
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
                    id_to_label(cast(int, l))
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
) -> Dict[str, float]:
    prediction, solution = compute_prediction_and_solution(p)

    iteration = len(iteration_tracking)

    iteration_result = evaluation.evaluate_iteration(
        run_name=run_name,
        iteration=iteration,
        average=average,
        solution=solution,
        result=prediction,
    )

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
) -> partial[Dict[str, float]]:
    return partial(
        compute_metrics,
        iteration_tracking=iteration_tracking,
        average=average,
        run_name=run_name,
    )


def create_tore_dataset(data: DataSet[DataDF]) -> Dataset:
    prepared_data = prepare_data(data)
    prepared_data_df = pd.DataFrame.from_records(prepared_data)
    return Dataset.from_pandas(df=prepared_data_df)
