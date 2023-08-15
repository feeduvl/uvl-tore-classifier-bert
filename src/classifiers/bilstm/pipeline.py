from collections.abc import Sequence
from typing import cast
from typing import List

import mlflow
import numpy as np
import tensorflow as tf
from strictly_typed_pandas.dataset import DataSet

from classifiers.bilstm import get_embeddings_and_categorical
from classifiers.bilstm import get_glove_model
from classifiers.bilstm import get_model
from classifiers.bilstm import get_result_df
from classifiers.bilstm import get_sentence_length
from classifiers.bilstm import MultiClassPrecision
from classifiers.bilstm import MultiClassRecall
from classifiers.bilstm import reverse_one_hot_encoding
from classifiers.bilstm.files import model_path
from tooling import evaluation
from tooling.config import BiLSTMConfig
from tooling.loading import import_dataset
from tooling.logging import logging_setup
from tooling.model import get_id2label
from tooling.model import get_label2id
from tooling.model import get_sentence_lengths
from tooling.model import Label_None_Pad
from tooling.model import PAD, ZERO
from tooling.model import ResultDF
from tooling.observability import end_tracing
from tooling.observability import log_artifacts
from tooling.sampling import DATA_TEST
from tooling.sampling import DATA_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset_k_fold
from tooling.transformation import get_class_weights
from tooling.transformation import transform_dataset
from tooling.types import IterationResult

logging = logging_setup(__name__)


def bilstm_pipeline(cfg: BiLSTMConfig, run_name: str) -> None:
    # Import Dataset
    import_dataset(cfg, run_name)

    # Transform Dataset
    transformed_dataset = transform_dataset(
        cfg, run_name, fill_with_zeros=True
    )

    # Get model constants
    sentence_length = get_sentence_length(
        cfg.bilstm, data=transformed_dataset["dataset"]
    )

    padded_labels: Sequence[Label_None_Pad] = list(
        set(transformed_dataset["labels"] + [ZERO])
    )
    mlflow.log_param("padded_labels", padded_labels)

    weights = cast(
        List[np.float32],
        get_class_weights(cfg.bilstm, transformed_dataset["dataset"]).tolist(),
    )
    weights.append(np.float32(1.0))  # account for padding label
    class_weights = {idx: value for idx, value in enumerate(weights)}

    label2id = get_label2id(padded_labels)
    id2label = get_id2label(padded_labels)
    mlflow.log_param("id2label", id2label)
    mlflow.log_param("label2id", label2id)

    # Download Model
    glove_model = get_glove_model()

    # Prepare evaluation tracking
    iteration_tracking: List[IterationResult] = []

    # Start kfold
    for iteration, dataset_paths in split_dataset_k_fold(
        name=run_name,
        dataset=transformed_dataset["dataset"],
        cfg_experiment=cfg.experiment,
    ):
        logging.info(f"Starting {iteration=}")
        # Load training data and create a training file
        data_train = load_split_dataset(
            name=run_name, filename=DATA_TRAIN, iteration=iteration
        )

        processed_data = get_embeddings_and_categorical(
            dataset=data_train,
            sentence_length=sentence_length,
            labels=padded_labels,
            glove_model=glove_model,
            label2id=label2id,
        )

        data_test = load_split_dataset(
            name=run_name, filename=DATA_TEST, iteration=iteration
        )

        processed_data_test = get_embeddings_and_categorical(
            dataset=data_test,
            sentence_length=sentence_length,
            labels=padded_labels,
            glove_model=glove_model,
            label2id=label2id,
        )

        # Get Model
        model = get_model(
            n_tags=len(set(padded_labels)),
            sentence_length=sentence_length,
            cfg_bilstm=cfg.bilstm,
            id2label=id2label,
            average=cfg.experiment.average,
        )

        # Train
        model.fit(
            x=np.array(processed_data["embeddings"]),
            y=np.array(processed_data["onehot_encoded"]),
            batch_size=cfg.bilstm.batch_size,
            epochs=cfg.bilstm.number_epochs,
            validation_data=(
                np.array(processed_data_test["embeddings"]),
                np.array(processed_data_test["onehot_encoded"]),
            ),
            class_weight=class_weights,
            verbose=cfg.bilstm.verbose,
        )

        model.save(model_path(name=run_name, iteration=iteration))
        log_artifacts(model_path(name=run_name, iteration=iteration))
        # Classify

        trained_model = tf.keras.models.load_model(
            model_path(name=run_name, iteration=iteration),
            compile=False,  # https://github.com/tensorflow/tensorflow/issues/31850#issuecomment-578566637
            custom_objects={
                "MultiClassPrecision": MultiClassPrecision,
                "MultiClassRecall": MultiClassRecall,
            },
        )

        categorical_predictions = trained_model.predict(
            processed_data_test["embeddings"]
        )

        # Create Solution
        solution = cast(DataSet[ResultDF], data_test[["string", "tore_label"]])

        # Evaluate Iteration

        label_df = reverse_one_hot_encoding(
            categorical_data=categorical_predictions,
            sentence_lengths=get_sentence_lengths(data_test),
            id2label=id2label,
        )

        result = get_result_df(dataset=data_test, label_df=label_df)

        evaluation.evaluate_iteration(
            run_name=run_name,
            iteration=iteration,
            average=cfg.experiment.average,
            solution=solution,
            result=result,
            labels=transformed_dataset["labels"],
            iteration_tracking=iteration_tracking,
        )

        logging.info(f"Finished {iteration=}")

        # early break if configured
        if iteration + 1 == cfg.experiment.iterations:
            logging.info(
                f"Breaking early after {iteration=} of {cfg.experiment.folds} folds"
            )
            break

    # Evaluate Run
    evaluation.evaluate_experiment(
        run_name=run_name, iteration_results=iteration_tracking
    )

    end_tracing()
