import numpy as np
from sklearn.model_selection import train_test_split
from typing import List
import pandas as pd
from pathlib import Path
from data import create_file
from typing import Tuple, cast
from data import EVALUATION_TEMP, evaluation_filepath


from sklearn.metrics import (
    ConfusionMatrixDisplay,
    recall_score,
    precision_score,
)

from dvclive import Live

import matplotlib.pyplot as plt


def create_live(name: str):
    return Live(evaluation_filepath(name=name, filename=""), dvcyaml=False)


def confusion_matrix(
    name: str,
    live: Live,
    solution: pd.Series,
    results: pd.Series,
):
    ConfusionMatrixDisplay.from_predictions(
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

    live.log_image("conf_matrix.png", fig_path)


def score_precision(
    live: Live, solution: pd.Series, results: pd.Series, labels: List[str]
):
    precision = precision_score(
        solution,
        results,
        average="macro",
        labels=labels,
        zero_division=0,
    )
    live.summary["precision"] = precision


def score_recall(
    live: Live, solution: pd.Series, results: pd.Series, labels: List[str]
):
    recall = recall_score(
        solution,
        results,
        average="macro",
        labels=labels,
        zero_division=0,
    )
    live.summary["recall"] = recall


def summarize(live: Live):
    live.make_summary()
