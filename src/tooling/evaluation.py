from pathlib import Path
from typing import cast
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data import create_file
from data import evaluation_filepath
from data import EVALUATION_TEMP
from sklearn import metrics
from sklearn.model_selection import train_test_split
from strictly_typed_pandas import DataSet
from tooling.model import Label
from tooling.model import ResultDF


def output_confusion_matrix(
    fig_path: Path,
    solution: DataSet[ResultDF],
    results: DataSet[ResultDF],
) -> None:
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=solution["tore_label"],
        y_pred=results["tore_label"],
        xticks_rotation="vertical",
        normalize=None,
        values_format=".2f",
    )

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


def score_precision(
    solution: DataSet[ResultDF],
    results: DataSet[ResultDF],
    labels: List[Label],
    average: str | None,
) -> float:
    precision: float = metrics.precision_score(
        solution["tore_label"],
        results["tore_label"],
        average=average,
        labels=labels,
        zero_division=0,
    )

    return precision


def score_recall(
    solution: DataSet[ResultDF],
    results: DataSet[ResultDF],
    labels: List[Label],
    average: str | None,
) -> float:
    recall: float = metrics.recall_score(
        solution["tore_label"],
        results["tore_label"],
        average=average,
        labels=labels,
        zero_division=0,
    )
    return recall
