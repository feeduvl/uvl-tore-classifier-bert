from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import TypedDict

import gensim.downloader as api
import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from strictly_typed_pandas import DataSet

from tooling.config import BiLSTM
from tooling.logging import logging_setup
from tooling.model import data_to_list_of_label_lists
from tooling.model import data_to_list_of_token_lists
from tooling.model import DataDF
from tooling.model import get_sentence_lengths
from tooling.model import HintedDataDF
from tooling.model import Label_None_Pad
from tooling.model import PAD
from tooling.model import ResultDF
from tooling.model import ToreLabelDF


logging = logging_setup(__name__)


def classify_with_bilstm(
    model_path: Path,
    data: DataSet[DataDF],
    max_len: int,
    glove_model: pd.DataFrame,
    hint_id2label: Dict[int, Label_None_Pad],
) -> DataSet[HintedDataDF]:
    embeddings = get_word_embeddings(
        dataset=data,
        glove_model=glove_model,
        sentence_length=max_len,
    )

    trained_model = tf.keras.models.load_model(
        model_path,
        compile=False,  # https://github.com/tensorflow/tensorflow/issues/31850#issuecomment-578566637
        custom_objects={
            "MultiClassPrecision": MultiClassPrecision,
            "MultiClassRecall": MultiClassRecall,
        },
    )

    categorical_predictions = trained_model.predict(embeddings)

    label_df = reverse_one_hot_encoding(
        categorical_data=categorical_predictions,
        sentence_lengths=get_sentence_lengths(data),
        id2label=hint_id2label,
    )

    data.reset_index(drop=True, inplace=True)
    label_df.reset_index(drop=True, inplace=True)
    data["hint_input_ids"] = label_df["tore_label"]

    return cast(DataSet[HintedDataDF], data)


def construct_model(n_tags: int, sentence_length: int) -> tf.keras.Model:
    input = tf.keras.Input(shape=(sentence_length, 100))
    model = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=100,
            return_sequences=True,
            input_shape=(sentence_length, 100),
        )
    )(input)
    model = tf.keras.layers.Dropout(0.1)(model)
    model = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=50,
            return_sequences=True,
        )
    )(model)
    out = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(n_tags, activation="softmax")
    )(
        model
    )  # softmax output layer

    return tf.keras.Model(input, out)


def compile_model(
    model: tf.keras.Model,
    cfg: BiLSTM,
    id2label: Dict[int, Label_None_Pad],
    average: str,
) -> None:
    model.compile(
        run_eagerly=True,
        optimizer=tf.keras.optimizers.legacy.Adam(
            learning_rate=cfg.learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=[
            MultiClassPrecision(  # type: ignore
                name="precision",
                label_count=len(id2label) - 1,
                average=average,
            ),
            MultiClassRecall(  # type: ignore
                name="recall",
                label_count=len(id2label) - 1,
                average=average,
            ),
        ],
    )


def get_model(
    n_tags: int,
    sentence_length: int,
    cfg_bilstm: BiLSTM,
    id2label: Dict[int, Label_None_Pad],
    average: str,
) -> tf.keras.Model:
    model = construct_model(n_tags=n_tags, sentence_length=sentence_length)
    compile_model(
        model=model,
        cfg=cfg_bilstm,
        id2label=id2label,
        average=average,
    )
    return model


def get_glove_model() -> pd.DataFrame:
    return cast(pd.DataFrame, api.load("glove-twitter-100"))


def get_one_hot_encoding(
    dataset: DataSet[DataDF],
    labels: Sequence[Label_None_Pad],
    sentence_length: int,
    label2id: Dict[Label_None_Pad, int],
) -> npt.NDArray[np.int32]:
    sentences_label_id_list = data_to_list_of_label_lists(
        data=dataset, label2id=label2id
    )

    padded_sent_label_id_list = pad_sequences(
        sequences=sentences_label_id_list,
        maxlen=sentence_length,
        padding="post",
        value=label2id[PAD],
    )

    return cast(
        npt.NDArray[np.int32],
        to_categorical(padded_sent_label_id_list, num_classes=len(labels)),
    )


def reverse_sentence_one_hot_encoding(
    categorical_data: npt.NDArray[np.int32] | npt.NDArray[np.float32],
    sentence_length: int,
    id2label: Dict[int, Label_None_Pad],
) -> List[Label_None_Pad]:
    padded_label_id_list: npt.NDArray[np.int32] = np.argmax(
        categorical_data, axis=-1
    )

    label_id_list = padded_label_id_list[:sentence_length]

    label_list: List[Label_None_Pad] = []
    for label_id in label_id_list:
        label_list.append(id2label[label_id])

    return label_list


def reverse_one_hot_encoding(
    categorical_data: npt.NDArray[np.int32] | npt.NDArray[np.float32],
    sentence_lengths: List[int],
    id2label: Dict[int, Label_None_Pad],
) -> DataSet[ToreLabelDF]:
    label_list: List[Label_None_Pad] = []

    for sentence_categorical_data, sentence_length in zip(
        categorical_data, sentence_lengths, strict=True
    ):
        label_list += reverse_sentence_one_hot_encoding(
            categorical_data=cast(
                npt.NDArray[np.float32], sentence_categorical_data
            ),
            sentence_length=sentence_length,
            id2label=id2label,
        )

    label_df = cast(
        DataSet[ToreLabelDF], pd.DataFrame(label_list, columns=["tore_label"])
    )

    return label_df


def get_result_df(
    dataset: DataSet[DataDF], label_df: DataSet[ToreLabelDF]
) -> DataSet[ResultDF]:
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


class ProcessedData(TypedDict):
    onehot_encoded: npt.NDArray[np.int32]
    embeddings: List[List[Any]]


def get_embeddings_and_categorical(
    dataset: DataSet[DataDF],
    sentence_length: int,
    labels: Sequence[Label_None_Pad],
    glove_model: pd.DataFrame,
    label2id: Dict[Label_None_Pad, int],
) -> ProcessedData:
    one_hot = get_one_hot_encoding(
        dataset=dataset,
        sentence_length=sentence_length,
        labels=labels,
        label2id=label2id,
    )

    embeddings = get_word_embeddings(
        dataset=dataset,
        glove_model=glove_model,
        sentence_length=sentence_length,
    )

    return ProcessedData(onehot_encoded=one_hot, embeddings=embeddings)


def get_sentence_length(bilstm_config: BiLSTM, data: DataSet[DataDF]) -> int:
    if bilstm_config.sentence_length is None:
        sentences_token_list = data_to_list_of_token_lists(data=data)
        sentence_length = max(
            [len(sentence_tl) for sentence_tl in sentences_token_list]
        )
        mlflow.log_param("computed_sentence_length", sentence_length)
        logging.info(f"Computed maximal sentence length: {sentence_length = }")
    else:
        sentence_length = bilstm_config.sentence_length
        mlflow.log_param("sentence_length", sentence_length)
        logging.info(
            f"Configured maximal sentence length: {sentence_length = }"
        )

    return sentence_length


class MultiClassPrecision(tf.keras.metrics.Metric):
    def __init__(  # type: ignore
        self, label_count, average=None, name="mc_precision", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.average = average
        self.label_count = label_count
        self.true_positives = self.add_weight(
            name="tp",
            initializer="zeros",
            shape=(label_count,),
            dtype=tf.int64,
        )
        self.false_positives = self.add_weight(
            name="fp",
            initializer="zeros",
            shape=(label_count,),
            dtype=tf.int64,
        )

    def get_config(self):  # type: ignore
        """Returns the serializable config of the metric."""
        return {
            "name": self.name,
            "label_count": self.label_count,
            "average": self.average,
        }

    @classmethod
    def from_config(cls, config):  # type: ignore
        """Creates a layer from its config.

        This method is the reverse of `get_config`,
        capable of instantiating the same layer from the config
        dictionary. It does not handle layer connectivity
        (handled by Network), nor weights (handled by `set_weights`).

        Args:
            config: A Python dictionary, typically the
                output of get_config.

        Returns:
            A layer instance.
        """
        try:
            return cls(**config)
        except Exception as e:
            raise TypeError(
                f"Error when deserializing class '{cls.__name__}' using "
                f"config={config}.\n\nException encountered: {e}"
            )

    def update_state(self, y_true, y_pred, sample_weight=None):  # type: ignore
        labels = tf.range(self.label_count, delta=1, dtype=tf.int64)

        y_true = tf.argmax(y_true, axis=-1)

        true_positives = tf.map_fn(
            fn=lambda t: tf.equal(y_true, t), elems=labels, dtype=tf.bool
        )

        y_pred = tf.argmax(y_pred, axis=-1)

        pred_positives = tf.map_fn(
            fn=lambda t: tf.equal(y_pred, t), elems=labels, dtype=tf.bool
        )

        tp = tf.logical_and(pred_positives, true_positives)
        tp = tf.math.count_nonzero(tp, axis=-1)

        inverted_true_positives = tf.logical_not(true_positives)
        fp = tf.logical_and(pred_positives, inverted_true_positives)
        fp = tf.math.count_nonzero(fp, axis=-1)

        self.true_positives.assign_add(tf.reduce_sum(tp, axis=-1))
        self.false_positives.assign_add(tf.reduce_sum(fp, axis=-1))

    def result(self):  # type: ignore
        if self.average == "micro":
            tp = tf.reduce_sum(self.true_positives)
            fp = tf.reduce_sum(self.false_positives)
            sum = tf.add(tp, fp)

            return tf.math.divide_no_nan(tp, sum)

        if self.average == "macro":
            tp = self.true_positives
            fp = self.false_positives
            sum = tf.add(tp, fp)
            recall = tf.math.divide_no_nan(tp, sum)
            return tf.reduce_mean(recall)

        if self.average is None:
            tp = self.true_positives
            fp = self.false_positives
            sum = tf.add(tp, fp)
            return tf.math.divide_no_nan(tp, sum)

    def reset_state(self):  # type: ignore
        self.true_positives.assign(
            tf.zeros(self.true_positives.shape, dtype=tf.int64)
        )
        self.false_positives.assign(
            tf.zeros(self.false_positives.shape, dtype=tf.int64)
        )


class MultiClassRecall(tf.keras.metrics.Metric):
    def __init__(  # type: ignore
        self, label_count, average=None, name="mc_precision", **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.label_count = label_count
        self.average = average
        self.true_positives = self.add_weight(
            name="tp",
            initializer="zeros",
            shape=(label_count,),
            dtype=tf.int64,
        )
        self.false_negatives = self.add_weight(
            name="fp",
            initializer="zeros",
            shape=(label_count,),
            dtype=tf.int64,
        )

    def get_config(self):  # type: ignore
        """Returns the serializable config of the metric."""
        return {
            "name": self.name,
            "label_count": self.label_count,
            "average": self.average,
        }

    @classmethod
    def from_config(cls, config):  # type: ignore
        """Creates a layer from its config.

        This method is the reverse of `get_config`,
        capable of instantiating the same layer from the config
        dictionary. It does not handle layer connectivity
        (handled by Network), nor weights (handled by `set_weights`).

        Args:
            config: A Python dictionary, typically the
                output of get_config.

        Returns:
            A layer instance.
        """
        try:
            return cls(**config)
        except Exception as e:
            raise TypeError(
                f"Error when deserializing class '{cls.__name__}' using "
                f"config={config}.\n\nException encountered: {e}"
            )

    def update_state(self, y_true, y_pred, sample_weight=None):  # type: ignore
        labels = tf.range(self.label_count, delta=1, dtype=tf.int64)

        y_true = tf.argmax(y_true, axis=-1)

        true_positives = tf.map_fn(
            fn=lambda t: tf.equal(y_true, t), elems=labels, dtype=tf.bool
        )

        y_pred = tf.argmax(y_pred, axis=-1)

        pred_positives = tf.map_fn(
            fn=lambda t: tf.equal(y_pred, t), elems=labels, dtype=tf.bool
        )

        tp = tf.logical_and(pred_positives, true_positives)
        tp = tf.math.count_nonzero(tp, axis=-1)

        inverted_pred_positives = tf.logical_not(pred_positives)
        fn = tf.logical_and(inverted_pred_positives, true_positives)
        fn = tf.math.count_nonzero(fn, axis=-1)

        self.true_positives.assign_add(tf.reduce_sum(tp, axis=-1))
        self.false_negatives.assign_add(tf.reduce_sum(fn, axis=-1))

    def result(self):  # type: ignore
        if self.average == "micro":
            tp = tf.reduce_sum(self.true_positives)
            fn = tf.reduce_sum(self.false_negatives)
            sum = tf.add(tp, fn)

            return tf.math.divide_no_nan(tp, sum)

        if self.average == "macro":
            tp = self.true_positives
            fn = self.false_negatives
            sum = tf.add(tp, fn)
            recall = tf.math.divide_no_nan(tp, sum)
            return tf.reduce_mean(recall)

        if self.average is None:
            tp = self.true_positives
            fn = self.false_negatives
            sum = tf.add(tp, fn)
            return tf.math.divide_no_nan(tp, sum)

    def reset_state(self):  # type: ignore
        self.true_positives.assign(
            tf.zeros(self.true_positives.shape, dtype=tf.int64)
        )
        self.false_negatives.assign(
            tf.zeros(self.false_negatives.shape, dtype=tf.int64)
        )
