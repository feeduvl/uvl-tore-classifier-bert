from pathlib import Path
from typing import cast
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import overload

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from strictly_typed_pandas.dataset import DataSet

from data import evaluation_filepath
from tooling.logging import logging_setup
from tooling.model import get_labels
from tooling.model import Label
from tooling.model import Label_None
from tooling.model import Label_None_Pad
from tooling.model import ResultDF
from tooling.model import ToreLabelDF
from tooling.observability import log_experiment_result
from tooling.observability import log_iteration_result
from tooling.types import ExperimentResult
from tooling.types import IterationResult

logging = logging_setup(__name__)


def output_confusion_matrix(
    fig_path: Path,
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF],
    results: DataSet[ResultDF] | DataSet[ToreLabelDF],
) -> None:
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=solution["tore_label"],
        y_pred=results["tore_label"],
        xticks_rotation="vertical",
        normalize="true",
        values_format=".2f",
    )
    plt.ioff()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def confusion_matrix(
    name: str,
    iteration: int,
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF],
    results: DataSet[ResultDF] | DataSet[ToreLabelDF],
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
    name: str,
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF],
    results: DataSet[ResultDF] | DataSet[ToreLabelDF],
) -> Path:
    fig_path = evaluation_filepath(
        name=name, filename=("./confusion_matrix.png")
    )
    output_confusion_matrix(
        fig_path=fig_path,
        solution=solution,
        results=results,
    )

    return fig_path


@overload
def score_precision(
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF],
    results: DataSet[ResultDF] | DataSet[ToreLabelDF],
    labels: List[Label_None],
    average: None,
) -> List[float]:
    ...


@overload
def score_precision(
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF],
    results: DataSet[ResultDF] | DataSet[ToreLabelDF],
    labels: List[Label_None],
    average: str,
) -> float:
    ...


def score_precision(
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF],
    results: DataSet[ResultDF] | DataSet[ToreLabelDF],
    labels: List[Label_None],
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
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF],
    results: DataSet[ResultDF] | DataSet[ToreLabelDF],
    labels: List[Label_None],
    average: None,
) -> List[float]:
    ...


@overload
def score_recall(
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF],
    results: DataSet[ResultDF] | DataSet[ToreLabelDF],
    labels: List[Label_None],
    average: str,
) -> float:
    ...


def score_recall(
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF],
    results: DataSet[ResultDF] | DataSet[ToreLabelDF],
    labels: List[Label_None],
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


def compute_f1(precision: float, recall: float) -> float:
    return 2 * ((precision * recall) / (precision + recall))


@overload
def evaluate(
    run_name: str,
    iteration: int,
    average: str,
    labels: List[Label_None],
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF],
    result: DataSet[ResultDF] | DataSet[ToreLabelDF],
    create_confusion_matrix: bool,
) -> IterationResult:
    ...


@overload
def evaluate(
    run_name: None,
    iteration: None,
    average: str,
    labels: List[Label_None],
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF],
    result: DataSet[ResultDF] | DataSet[ToreLabelDF],
    create_confusion_matrix: Literal[False],
) -> IterationResult:
    ...


def evaluate(
    run_name: Optional[str],
    iteration: Optional[int],
    average: str,
    labels: List[Label_None],
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF],
    result: DataSet[ResultDF] | DataSet[ToreLabelDF],
    create_confusion_matrix: bool = True,
) -> IterationResult:
    res = IterationResult(step=iteration, result=result, solution=solution)

    # all should-be-contained-labels
    # solution_labels = get_labels(solution)
    # res.label_count = len(solution_labels)
    solution_labels = labels
    res.label_count = len(labels)

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
    res.f1 = compute_f1(precision=res.precision, recall=res.recall)

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

    if create_confusion_matrix and run_name and iteration:
        res.confusion_matrix = confusion_matrix(
            name=run_name,
            iteration=iteration,
            solution=solution,
            results=result,
        )

    return res


def evaluate_iteration(
    run_name: str,
    iteration: int,
    average: str,
    labels: List[Label_None],
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF],
    result: DataSet[ResultDF] | DataSet[ToreLabelDF],
    iteration_tracking: List[IterationResult],
    create_confusion_matrix: bool = True,
) -> IterationResult:
    res = evaluate(
        run_name=run_name,
        iteration=iteration,
        average=average,
        labels=labels,
        solution=solution,
        result=result,
        create_confusion_matrix=create_confusion_matrix,
    )
    iteration_tracking.append(res)

    log_iteration_result(res)
    logging.info(f"Logged iteration result {res.precision=} {res.recall=}")

    return res


def evaluate_experiment(
    run_name: str,
    iteration_results: List[IterationResult],
) -> ExperimentResult:
    res = ExperimentResult()
    results_df = pd.DataFrame(iteration_results)

    res.label_count = results_df["label_count"].max()

    res.min_f1 = results_df["f1"].min()
    res.mean_f1 = results_df["f1"].mean()
    res.max_f1 = results_df["f1"].max()

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

    log_experiment_result(result=res)
    logging.info(
        f"Logged experiment result {res.mean_precision=} {res.mean_recall=}"
    )

    return res
