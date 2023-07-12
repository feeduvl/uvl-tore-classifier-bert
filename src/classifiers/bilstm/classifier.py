from collections.abc import Sequence
from typing import Any
from typing import cast
from typing import List

import gensim.downloader as api
import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from strictly_typed_pandas import DataSet

from tooling.model import data_to_list_of_label_lists
from tooling.model import data_to_list_of_token_lists
from tooling.model import DataDF
from tooling.model import Label_None_Pad
from tooling.model import ResultDF


def construct_model(n_tags: int, sentence_length: int) -> tf.keras.Model:
    input = tf.keras.Input(shape=(sentence_length, 100))
    model = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=100,
            return_sequences=True,
            recurrent_dropout=0.1,
            input_shape=(sentence_length, 100),
        )
    )(input)
    model = tf.keras.layers.Dropout(0.1)(model)
    model = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=50, return_sequences=True, recurrent_dropout=0.1
        )
    )(model)
    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_tags, activation="softmax")
    )(
        model
    )  # softmax output layer

    return tf.keras.Model(input, out)


def get_glove_model() -> pd.DataFrame:
    return cast(pd.DataFrame, api.load("glove-twitter-100"))


def get_one_hot_encoding(
    dataset: DataSet[DataDF],
    labels: Sequence[Label_None_Pad],
    sentence_length: int,
) -> npt.NDArray[np.int32]:
    sentences_label_list = data_to_list_of_label_lists(data=dataset)

    sentences_label_id_list: List[List[int]] = []
    for sentence_tl in sentences_label_list:
        sentence_token_id_list: List[int] = []
        for token_tore_label in sentence_tl:
            sentence_token_id_list.append(labels.index(token_tore_label))
        sentences_label_id_list.append(sentence_token_id_list)

    padded_sent_label_id_list = pad_sequences(
        sequences=sentences_label_id_list,
        maxlen=sentence_length,
        padding="post",
        value=labels.index("0"),
    )

    return cast(
        npt.NDArray[np.int32],
        to_categorical(padded_sent_label_id_list, num_classes=len(labels)),
    )


def reverse_sentence_one_hot_encoding(
    categorical_data: npt.NDArray[np.int32] | npt.NDArray[np.float32],
    labels: Sequence[Label_None_Pad],
    sentence_length: int,
) -> List[Label_None_Pad]:
    padded_label_id_list: npt.NDArray[np.int32] = np.argmax(
        categorical_data, axis=-1
    )

    label_id_list = padded_label_id_list[:sentence_length]

    label_list: List[Label_None_Pad] = []
    for label_id in label_id_list:
        label_list.append(labels[label_id])

    return label_list


def reverse_one_hot_encoding(
    dataset: DataSet[DataDF],
    categorical_data: npt.NDArray[np.int32] | npt.NDArray[np.float32],
    labels: Sequence[Label_None_Pad],
) -> DataSet[ResultDF]:
    label_list: List[Label_None_Pad] = []

    actual_sentence_lengths = [
        len(token_list) for token_list in data_to_list_of_token_lists(dataset)
    ]

    for sentence_categorical_data, actual_sentence_length in zip(
        categorical_data, actual_sentence_lengths, strict=True
    ):
        label_list += reverse_sentence_one_hot_encoding(
            categorical_data=cast(
                npt.NDArray[np.float32], sentence_categorical_data
            ),
            labels=labels,
            sentence_length=actual_sentence_length,
        )

    label_df = pd.DataFrame(label_list, columns=["tore_label"])

    dataset.reset_index(inplace=True)
    label_df.reset_index(inplace=True)

    result_df = cast(
        DataSet[ResultDF],
        pd.concat(
            [dataset["string"], label_df["tore_label"]],
            axis="columns",
        ),
    )
    return result_df


def pad_or_truncate(some_list: List[Any], target_len: int) -> List[Any]:
    return some_list[:target_len] + [np.zeros(100).tolist()] * (
        target_len - len(some_list)
    )


def get_word_embeddings(
    dataset: DataSet[DataDF],
    glove_model: pd.DataFrame,
    sentence_length: int,
) -> List[List[Any]]:
    sentences_token_list = data_to_list_of_token_lists(data=dataset)

    word_embeddings = [
        [
            (
                glove_model[word].tolist()
                if (word in glove_model)
                else np.zeros(100).tolist()
            )
            for word in sentence
        ]
        for sentence in sentences_token_list
    ]
    sized_word_embeddings = [
        pad_or_truncate(s, sentence_length) for s in word_embeddings
    ]

    return sized_word_embeddings
