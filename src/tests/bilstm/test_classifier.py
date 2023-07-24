import uuid
from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from strictly_typed_pandas import DataSet

from classifiers.bilstm import get_one_hot_encoding
from classifiers.bilstm import get_result_df
from classifiers.bilstm import MultiClassPrecision
from classifiers.bilstm import MultiClassRecall
from classifiers.bilstm import reverse_one_hot_encoding
from tooling.model import DataDF
from tooling.model import get_id2label
from tooling.model import get_label2id
from tooling.model import get_sentence_lengths
from tooling.model import Label_None_Pad
from tooling.model import Token
from tooling.model import tokenlist_to_datadf
from tooling.model import TORE_LABELS_NONE_PAD


def test_one_hot_encoding() -> None:
    sentences = []
    label_selection: Sequence[Label_None_Pad] = TORE_LABELS_NONE_PAD
    generation_sentence_length = 10

    dfs = []

    label2id = get_label2id(label_selection)
    id2label = get_id2label(label_selection)

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
            df,
            labels=label_selection,
            sentence_length=sentence_length,
            label2id=label2id,
        )
        sentences.append(one_hot)
        dfs.append(df)

    complete_df = cast(DataSet[DataDF], pd.concat(dfs, ignore_index=True))

    categorical_data = np.concatenate(sentences, axis=0)

    reversed_one_hot = reverse_one_hot_encoding(
        categorical_data=categorical_data,
        sentence_lengths=get_sentence_lengths(complete_df),
        id2label=id2label,
    )

    result_df = get_result_df(dataset=complete_df, label_df=reversed_one_hot)

    pd.testing.assert_frame_equal(
        complete_df[["string", "tore_label"]], result_df
    )


def test_mc_precision() -> None:
    y_pred = tf.constant(
        [
            [[0, 0.7, 0.2], [0, 0.7, 0.2], [0, 0.2, 0.7]],
            [[0, 0.7, 0.2], [0, 0.7, 0.2], [0, 0.2, 0.7]],
            [[0, 0.7, 0.2], [0, 0.7, 0.2], [0, 0.2, 0.7]],
        ]
    )
    y_true = tf.constant(
        [
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        ]
    )

    m = MultiClassPrecision(label_count=3)  # type: ignore
    m.update_state(y_pred=y_pred, y_true=y_true)  # type: ignore
    np.testing.assert_array_almost_equal(
        m.result().numpy(), np.array([0, 0, 1])  # type: ignore
    )

    m = MultiClassPrecision(label_count=3, average="micro")  # type: ignore
    m.update_state(y_pred=y_pred, y_true=y_true)  # type: ignore
    assert m.result().numpy() == pytest.approx(1 / 3)  # type: ignore

    m = MultiClassPrecision(label_count=3, average="macro")  # type: ignore
    m.update_state(y_pred=y_pred, y_true=y_true)  # type: ignore
    assert m.result().numpy() == pytest.approx(1 / 3)  # type: ignore

    m.reset_state()  # type: ignore


def test_mc_recall() -> None:
    y_pred = tf.constant(
        [
            [[0, 0.7, 0.2], [0, 0.7, 0.2], [0, 0.2, 0.7]],
            [[0, 0.7, 0.2], [0, 0.7, 0.2], [0, 0.2, 0.7]],
            [[0, 0.7, 0.2], [0, 0.7, 0.2], [0, 0.2, 0.7]],
        ]
    )
    y_true = tf.constant(
        [
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
        ]
    )

    m = MultiClassRecall(label_count=3)  # type: ignore
    m.update_state(y_pred=y_pred, y_true=y_true)  # type: ignore
    np.testing.assert_array_almost_equal(
        m.result().numpy(), np.array([0, 0, 1 / 3])  # type: ignore
    )

    m = MultiClassRecall(label_count=3, average="micro")  # type: ignore
    m.update_state(y_pred=y_pred, y_true=y_true)  # type: ignore
    assert m.result().numpy() == pytest.approx(1 / 3)  # type: ignore

    m = MultiClassRecall(label_count=3, average="macro")  # type: ignore
    m.update_state(y_pred=y_pred, y_true=y_true)  # type: ignore
    assert m.result().numpy() == pytest.approx(1 / 9)  # type: ignore

    m.reset_state()  # type: ignore
