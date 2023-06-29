from typing import Any
from typing import Dict
from typing import List

import gensim.downloader as api
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from strictly_typed_pandas import DataSet

from tooling.model import data_to_list_of_label_lists
from tooling.model import data_to_list_of_token_lists
from tooling.model import DataDF
from tooling.model import get_labels


def get_glove_model() -> Any:
    return api.load("glove-twitter-100")


def get_one_hot_encoding(
    dataset: DataSet[DataDF], sentence_length: int | None = None
) -> List[Any]:
    sentences_label_list = data_to_list_of_label_lists(data=dataset)
    labels = get_labels(dataset=dataset)

    if sentence_length is None:
        sentence_length = max(
            [len(sentence_tl) for sentence_tl in sentences_label_list]
        )

    sentences_label_id_list: List[List[int]]
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

    categorical_sent_label_id_l = [
        to_categorical(padded_sent_label_id, num_classes=len(labels))
        for padded_sent_label_id in padded_sent_label_id_list
    ]

    return categorical_sent_label_id_l


def pad_or_truncate(some_list: List[Any], target_len: int) -> List[Any]:
    return some_list[:target_len] + [np.zeros(100).tolist()] * (
        target_len - len(some_list)
    )


def get_word_embeddings(
    dataset: DataSet[DataDF],
    glove_model: Dict[str, pd.Dataframe],
    sentence_length: int | None = None,
):
    sentences_token_list = data_to_list_of_token_lists(data=dataset)

    if sentence_length is None:
        sentence_length = max(
            [len(sentence_tl) for sentence_tl in sentences_token_list]
        )

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
