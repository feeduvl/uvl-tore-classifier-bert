import pytest

import pandas as pd
from collections import Counter
from tooling.sampling import apply_smote

def test_apply_smote() -> None:
    test_dataset_smote_filepath = "test_dataset_smote.json"

    imbalanced_df = pd.read_json(test_dataset_smote_filepath)
    counts_imbalanced_df = dict(Counter(imbalanced_df["tore_label"]))

    assert(counts_imbalanced_df["Domain_Level"] == 12)
    assert(counts_imbalanced_df["Interaction_Level"] == 9)
    assert(counts_imbalanced_df["System_Level"] == 9)

    balanced_df = apply_smote(imbalanced_df, 5, "not majority", False)
    counts_balanced_df = dict(Counter(balanced_df["tore_label"]))

    assert(counts_balanced_df["Domain_Level"] == 12)
    assert(counts_balanced_df["Interaction_Level"] == 12)
    assert(counts_balanced_df["System_Level"] == 12)
    assert(list(imbalanced_df["tore_label"]) == list(balanced_df["tore_label"][:45]))
    assert(len(balanced_df["tore_label"]) == 51)


def test_apply_smote_balance_to_average() -> None:
    test_dataset_smote_filepath = "test_dataset_smote.json"

    imbalanced_df = pd.read_json(test_dataset_smote_filepath)
    counts_imbalanced_df = dict(Counter(imbalanced_df["tore_label"]))

    assert(counts_imbalanced_df["Domain_Level"], 12)
    assert(counts_imbalanced_df["Interaction_Level"], 9)
    assert(counts_imbalanced_df["System_Level"], 9)

    balanced_df = apply_smote(imbalanced_df, 5, "not majority", True)
    counts_balanced_df = dict(Counter(balanced_df["tore_label"]))

    assert(counts_balanced_df["Domain_Level"], 12)
    assert(counts_balanced_df["Interaction_Level"], 10)
    assert(counts_balanced_df["System_Level"], 10)
    assert(list(imbalanced_df["tore_label"]), list(balanced_df["tore_label"][:45]))
    assert(len(balanced_df["tore_label"]), 47)

