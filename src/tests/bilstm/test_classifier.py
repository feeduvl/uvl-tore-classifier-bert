import uuid
from collections.abc import Sequence
from typing import cast
from typing import get_args
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from strictly_typed_pandas import DataSet

from classifiers.bilstm import get_one_hot_encoding
from classifiers.bilstm import reverse_one_hot_encoding
from tooling.model import DataDF
from tooling.model import get_labels
from tooling.model import Label
from tooling.model import Token
from tooling.model import tokenlist_to_datadf
from tooling.model import TORE_LABELS_0


def test_one_hot_encoding() -> None:
    sentences = []
    label_selection: Sequence[Label] = TORE_LABELS_0
    generation_sentence_length = 10

    dfs = []

    for s in range(10):
        tokens = []

        sentence_id = uuid.uuid1()

        for i in range(generation_sentence_length):
            selected_label = label_selection[i]
            t = Token(
                sentence_id=sentence_id,
                sentence_idx=i,
                string="ABC",
                lemma="abc",
                pos="a",
                source="test",
                tore_label=selected_label,
            )
            tokens.append(t)

        sentence_length = 14
        df = tokenlist_to_datadf(tokens)
        one_hot = get_one_hot_encoding(
            df, labels=label_selection, sentence_length=sentence_length
        )
        sentences.append(one_hot)
        dfs.append(df)

    complete_df = cast(DataSet[DataDF], pd.concat(dfs, ignore_index=True))

    categorical_data = np.concatenate(sentences, axis=0)

    reversed_one_hot = reverse_one_hot_encoding(
        dataset=complete_df,
        categorical_data=categorical_data,
        labels=label_selection,
    )

    pd.testing.assert_frame_equal(
        complete_df[["string", "tore_label"]], reversed_one_hot
    )
