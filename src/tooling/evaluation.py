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


def output_confusion_matrix(
    fig_path: Path,
    solution: pd.Series,
    results: pd.Series,
) -> Path:
    metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=solution,
        y_pred=results,
        xticks_rotation="vertical",
        normalize="true",
        values_format=".2f",
    )

    plt.savefig(fig_path, bbox_inches="tight")


def confusion_matrix(
    name: str, iteration: int, solution: pd.Series, results: pd.Series
):
    fig_path = evaluation_filepath(
        name=name, filename=(f"./{iteration}_confusion_matrix.png")
    )
    output_confusion_matrix(
        fig_path=fig_path, solution=solution, results=results
    )

    return fig_path


def sum_confusion_matrix(name: str, solution: pd.Series, results: pd.Series):
    fig_path = evaluation_filepath(
        name=name, filename=(f"./confusion_matrix.png")
    )
    output_confusion_matrix(
        fig_path=fig_path, solution=solution, results=results
    )

    return fig_path


def score_precision(
    solution: pd.Series,
    results: pd.Series,
    labels: List[str],
    average: str | None,
) -> float:
    precision: float = metrics.precision_score(
        solution,
        results,
        average=average,
        labels=labels,
        zero_division=0,
    )

    return precision


def score_recall(
    solution: pd.Series,
    results: pd.Series,
    labels: List[str],
    average: str | None,
) -> float:
    recall: float = metrics.recall_score(
        solution,
        results,
        average=average,
        labels=labels,
        zero_division=0,
    )
    return recall
