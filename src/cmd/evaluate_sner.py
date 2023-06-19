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


results = load_classification_result()

labels_test = load_split_dataset(LABELS_TEST)
solution = sentences_to_token_df(pd.Series(labels_test))


precision = precision_score(
    solution["label"], results["label"], average="macro", labels=TORE_LABELS
)

recall = recall_score(
    solution["label"], results["label"], average="macro", labels=TORE_LABELS
)


# ConfusionMatrixDisplay.from_predictions()
LIVE_PATH = TEMP_PATH.joinpath(Path("eval/live"))
live = Live(LIVE_PATH, dvcyaml=False)

live.summary["precission"] = precision
live.summary["recall"] = recall

live.log_sklearn_plot(
    "confusion_matrix",
    solution["label"],
    results["label"],
    labels=TORE_LABELS,
    xticks_rotation="vertical",
)

live.make_summary()
