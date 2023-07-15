from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from strictly_typed_pandas import DataSet

from data import create_file
from data import evaluation_filepath
from data import EVALUATION_TEMP
from tooling.model import get_labels
from tooling.model import Label
from tooling.model import ResultDF


def output_confusion_matrix(
    fig_path: Path,
    solution: DataSet[ResultDF],
    results: DataSet[ResultDF],
) -> None:
    conf = metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=solution["tore_label"],
        y_pred=results["tore_label"],
        xticks_rotation="vertical",
        normalize="true",
        values_format=".2f",
    )
    plt.ioff()
    plt.savefig(fig_path, bbox_inches="tight")


def confusion_matrix(
    name: str,
    iteration: int,
    solution: DataSet[ResultDF],
    results: DataSet[ResultDF],
) -> Path:
    fig_path = evaluation_filepath(
        name=name, filename=(f"./{iteration}_confusion_matrix.png")
    )
    output_confusion_matrix(
        fig_path=fig_path,
        solution=solution,
        results=results,
    )

    return fig_path


def sum_confusion_matrix(
    name: str, solution: DataSet[ResultDF], results: DataSet[ResultDF]
) -> Path:
    fig_path = evaluation_filepath(
        name=name, filename=(f"./confusion_matrix.png")
    )
    output_confusion_matrix(
        fig_path=fig_path,
        solution=solution,
        results=results,
    )

    return fig_path


@overload
def score_precision(
    solution: DataSet[ResultDF],
    results: DataSet[ResultDF],
    labels: List[Label],
    average: None,
) -> List[float]:
    ...


@overload
def score_precision(
    solution: DataSet[ResultDF],
    results: DataSet[ResultDF],
    labels: List[Label],
    average: str,
) -> float:
    ...


def score_precision(
    solution: DataSet[ResultDF],
    results: DataSet[ResultDF],
    labels: List[Label],
    average: str | None,
) -> float | List[float]:
    precision: float = metrics.precision_score(
        solution["tore_label"],
        results["tore_label"],
        average=average,
        labels=labels,
        zero_division=0,
    )

    return precision


@overload
def score_recall(
    solution: DataSet[ResultDF],
    results: DataSet[ResultDF],
    labels: List[Label],
    average: None,
) -> List[float]:
    ...


@overload
def score_recall(
    solution: DataSet[ResultDF],
    results: DataSet[ResultDF],
    labels: List[Label],
    average: str,
) -> float:
    ...


def score_recall(
    solution: DataSet[ResultDF],
    results: DataSet[ResultDF],
    labels: List[Label],
    average: str | None,
) -> float | List[float]:
    recall: float = metrics.recall_score(
        solution["tore_label"],
        results["tore_label"],
        average=average,
        labels=labels,
        zero_division=0,
    )
    return recall


@dataclass
class IterationResult:
    step: int
    result: DataSet[ResultDF]
    solution: DataSet[ResultDF]

    precision: float = 0.0
    recall: float = 0.0

    pl_precision: Dict[Label, float] = field(default_factory=dict)
    pl_recall: Dict[Label, float] = field(default_factory=dict)

    confusion_matrix: Optional[Path] = None

    label_count: int = 0


def evaluate_iteration(
    run_name: str,
    iteration: int,
    average: str,
    solution: DataSet[ResultDF],
    result: DataSet[ResultDF],
    create_confusion_matrix: bool = True,
) -> IterationResult:
    res = IterationResult(step=iteration, result=result, solution=solution)

    # all should-be-contained-labels
    solution_labels = get_labels(solution)
    res.label_count = len(solution_labels)

    res.precision = score_precision(
        solution=solution,
        results=result,
        labels=solution_labels,
        average=average,
    )
    res.recall = score_recall(
        solution=solution,
        results=result,
        labels=solution_labels,
        average=average,
    )

    pl_precision = score_precision(
        solution=solution,
        results=result,
        labels=solution_labels,
        average=None,
    )

    res.pl_precision = dict(zip(solution_labels, pl_precision, strict=True))

    pl_recall = score_recall(
        solution=solution,
        results=result,
        labels=solution_labels,
        average=None,
    )

    res.pl_recall = dict(zip(solution_labels, pl_recall, strict=True))

    if create_confusion_matrix:
        res.confusion_matrix = confusion_matrix(
            name=run_name,
            iteration=iteration,
            solution=solution,
            results=result,
        )

    return res


@dataclass
class ExperimentResult:
    label_count: int = 0

    min_precision: float = 0.0
    min_recall: float = 0.0
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    max_precision: float = 0.0
    max_recall: float = 0.0

    pl_mean_precision: Dict[Label, float] = field(default_factory=dict)
    pl_mean_recall: Dict[Label, float] = field(default_factory=dict)

    confusion_matrix: Path = Path()


def evaluate_experiment(
    run_name: str,
    iteration_results: List[IterationResult],
) -> ExperimentResult:
    res = ExperimentResult()
    results_df = pd.DataFrame(iteration_results)

    res.label_count = results_df["label_count"].max()

    res.min_precision = results_df["precision"].min()
    res.mean_precision = results_df["precision"].mean()
    res.max_precision = results_df["precision"].max()

    res.min_recall = results_df["recall"].min()
    res.mean_recall = results_df["recall"].mean()
    res.max_recall = results_df["recall"].max()

    res.pl_mean_precision = (
        pd.DataFrame(list(results_df["pl_precision"])).mean(axis=0).to_dict()
    )

    res.pl_mean_recall = (
        pd.DataFrame(list(results_df["pl_recall"])).mean(axis=0).to_dict()
    )

    all_results = cast(
        DataSet[ResultDF], pd.concat(results_df["result"].to_list())
    )
    all_solutions = cast(
        DataSet[ResultDF], pd.concat(results_df["solution"].to_list())
    )
    res.confusion_matrix = sum_confusion_matrix(
        name=run_name,
        solution=all_solutions,
        results=all_results,
    )

    return res
