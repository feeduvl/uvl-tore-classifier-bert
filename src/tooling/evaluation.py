import numpy as np
from sklearn.model_selection import train_test_split
from typing import List
import pandas as pd
from pathlib import Path
from data import create_file
from typing import Tuple, cast
from data import EVALUATION_TEMP, evaluation_filepath
from sklearn import metrics
import matplotlib.pyplot as plt


def confusion_matrix(
    name: str,
    solution: pd.Series,
    results: pd.Series,
):
    metrics.ConfusionMatrixDisplay.from_predictions(
        solution,
        results,
        xticks_rotation="vertical",
        normalize="true",
        values_format=".2f",
    )

    fig_path = evaluation_filepath(
        name=name, filename=("./confusion_matrix.png")
    )

    plt.savefig(fig_path)


def score_precision(
    solution: pd.Series, results: pd.Series, labels: List[str]
):
    precision = metrics.precision_score(
        solution,
        results,
        average="macro",
        labels=labels,
        zero_division=0,
    )
    print(f"Precision: {precision}")
    return precision


def score_recall(solution: pd.Series, results: pd.Series, labels: List[str]):
    recall = metrics.recall_score(
        solution,
        results,
        average="macro",
        labels=labels,
        zero_division=0,
    )
    print(f"Recall: {recall}")
    return recall
