from classifiers.sner import (
    sentences_to_token_df,
    load_classification_result,
    TEMP_PATH,
)
from data import load_split_dataset, LABELS_TEST, TORE_LABELS
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    recall_score,
    precision_score,
)
import pandas as pd
from dvclive import Live
from pathlib import Path
import matplotlib.pyplot as plt

results = load_classification_result()

labels_test = load_split_dataset(LABELS_TEST)
solution = sentences_to_token_df(pd.Series(labels_test))


precision = precision_score(
    solution["label"],
    results["label"],
    average="macro",
    labels=TORE_LABELS,
    zero_division=0,
)

recall = recall_score(
    solution["label"],
    results["label"],
    average="macro",
    labels=TORE_LABELS,
    zero_division=0,
)


# ConfusionMatrixDisplay.from_predictions()
LIVE_PATH = TEMP_PATH.joinpath(Path("eval/live"))
live = Live(LIVE_PATH, dvcyaml=False)

live.summary["precission"] = precision
live.summary["recall"] = recall


CONF_MATRIX_PATH = LIVE_PATH.joinpath("./confusion_matrix.png")
ConfusionMatrixDisplay.from_predictions(
    solution["label"],
    results["label"],
    xticks_rotation="vertical",
    normalize="true",
    values_format=".2f",
)
plt.savefig(CONF_MATRIX_PATH)

live.log_image("conf_matrix.png", CONF_MATRIX_PATH)


live.make_summary()
