from unittest import TestCase
import pandas as pd
from collections import Counter
from tooling.sampling import apply_smote


class Test(TestCase):

    def test_apply_smote(self) -> None:
        test_dataset_smote_filepath = "test_dataset_smote.json"

        imbalanced_df = pd.read_json(test_dataset_smote_filepath)
        counts_imbalanced_df = dict(Counter(imbalanced_df["tore_label"]))

        self.assertEqual(counts_imbalanced_df["Domain_Level"], 12)
        self.assertEqual(counts_imbalanced_df["Interaction_Level"], 9)
        self.assertEqual(counts_imbalanced_df["System_Level"], 9)

        balanced_df = apply_smote(imbalanced_df)
        counts_balanced_df = dict(Counter(balanced_df["tore_label"]))

        self.assertEqual(counts_balanced_df["Domain_Level"], 12)
        self.assertEqual(counts_balanced_df["Interaction_Level"], 12)
        self.assertEqual(counts_balanced_df["System_Level"], 12)
        self.assertEqual(list(imbalanced_df["tore_label"]), list(balanced_df["tore_label"][:45]))
        self.assertEqual(len(balanced_df["tore_label"]), 51)