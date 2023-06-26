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


def confusion_matrix(
    name: str,
    solution: pd.Series,
    results: pd.Series,
) -> Path:
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
    return fig_path


def score_precision(
    solution: pd.Series, results: pd.Series, labels: List[str]
) -> float:
    precision: float = metrics.precision_score(
        solution,
        results,
        average="macro",
        labels=labels,
        zero_division=0,
    )
    print(f"Precision: {precision}")
    return precision


def score_recall(
    solution: pd.Series, results: pd.Series, labels: List[str]
) -> float:
    recall: float = metrics.recall_score(
        solution,
        results,
        average="macro",
        labels=labels,
        zero_division=0,
    )
    print(f"Recall: {recall}")
    return recall
