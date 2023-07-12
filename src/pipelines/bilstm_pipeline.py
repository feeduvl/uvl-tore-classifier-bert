from collections.abc import Sequence
from typing import cast
from typing import List

import hydra
import numpy as np
import tensorflow as tf
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from strictly_typed_pandas import DataSet

from classifiers.bilstm import construct_model
from classifiers.bilstm import get_glove_model
from classifiers.bilstm import get_one_hot_encoding
from classifiers.bilstm import get_word_embeddings
from classifiers.bilstm import reverse_one_hot_encoding
from classifiers.bilstm.files import model_path
from data import get_dataset_information
from tooling import evaluation
from tooling.config import BiLSTMConfig
from tooling.loading import import_dataset
from tooling.loading import load_dataset
from tooling.model import data_to_list_of_token_lists
from tooling.model import get_labels
from tooling.model import Label_None_Pad
from tooling.model import PAD
from tooling.model import ResultDF
from tooling.observability import config_mlflow
from tooling.observability import end_tracing
from tooling.observability import log_artifacts
from tooling.observability import log_experiment_result
from tooling.observability import log_iteration_result
from tooling.sampling import DATA_TEST
from tooling.sampling import DATA_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset_k_fold
from tooling.transformation import transform_dataset

tf.config.set_visible_devices([], "GPU")


cs = ConfigStore.instance()
cs.store(name="base_config", node=BiLSTMConfig)


@hydra.main(version_base=None, config_path="conf", config_name="epoch_sweep")
def main(cfg: BiLSTMConfig) -> None:
    # Setup experiment
    print(OmegaConf.to_yaml(cfg))
    run_name = config_mlflow(cfg)

    # Download Model
    glove_model = get_glove_model()

    # Import Dataset
    dataset_information = get_dataset_information(cfg.experiment.dataset)
    imported_dataset_path = import_dataset(
        name=run_name, ds_spec=dataset_information
    )
    log_artifacts(imported_dataset_path)

    # Transform Dataset
    d = load_dataset(name=run_name)
    dataset_labels, transformed_d = transform_dataset(
        dataset=d, cfg=cfg.transformation
    )

    transformed_d.fillna("0", inplace=True)

    sentence_length = cfg.bilstm.sentence_length
    if sentence_length is None:
        sentences_token_list = data_to_list_of_token_lists(data=transformed_d)
        sentence_length = max(
            [len(sentence_tl) for sentence_tl in sentences_token_list]
        )

    labels = get_labels(dataset=transformed_d)
    padded_labels: Sequence[Label_None_Pad] = labels + [PAD]

    iteration_tracking: List[evaluation.IterationResult] = []

    # Start kfold
    for iteration, dataset_paths in split_dataset_k_fold(
        name=run_name,
        dataset=transformed_d,
        folds=cfg.experiment.folds,
        random_state=cfg.experiment.random_state,
    ):
        log_artifacts(dataset_paths)

        # Load training data and create a training file
        data_train = load_split_dataset(
            name=run_name, filename=DATA_TRAIN, iteration=iteration
        )

        one_hot = get_one_hot_encoding(
            dataset=data_train,
            sentence_length=sentence_length,
            labels=padded_labels,
        )

        embeddings = get_word_embeddings(
            dataset=data_train,
            glove_model=glove_model,
            sentence_length=sentence_length,
        )

        # Configure Model
        model = construct_model(
            n_tags=len(padded_labels), sentence_length=sentence_length
        )

        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(),
            loss="categorical_crossentropy",
            metrics=["accuracy", tf.keras.metrics.MeanSquaredError()],
        )

        # Train
        history = model.fit(
            np.array(embeddings),
            np.array(one_hot),
            batch_size=cfg.bilstm.batch_size,
            epochs=cfg.bilstm.number_epochs,
            validation_split=cfg.bilstm.validation_split,
            verbose=cfg.bilstm.verbose,
        )

        model.save(model_path(name=run_name, iteration=iteration))
        model.summary()

        # Classify
        data_test = load_split_dataset(
            name=run_name, filename=DATA_TEST, iteration=iteration
        )

        embeddings_test = get_word_embeddings(
            dataset=data_test,
            glove_model=glove_model,
            sentence_length=sentence_length,
        )

        trained_model = tf.keras.models.load_model(
            model_path(name=run_name, iteration=iteration),
            compile=False,  # https://github.com/tensorflow/tensorflow/issues/31850#issuecomment-578566637
        )
        predictions_one_hot = trained_model.predict(embeddings_test)

        result = reverse_one_hot_encoding(
            dataset=data_test,
            categorical_data=predictions_one_hot,
            labels=labels,
        )

        # Create Solution
        solution = cast(DataSet[ResultDF], data_test[["string", "tore_label"]])

        # Evaluate Iteration
        iteration_result = evaluation.evaluate_iteration(
            run_name=run_name,
            iteration=iteration,
            average=cfg.experiment.average,
            solution=solution,
            result=result,
        )
        iteration_tracking.append(iteration_result)
        log_iteration_result(iteration_result)

    # Evaluate Run
    experiment_result = evaluation.evaluate_experiment(
        run_name=run_name, iteration_results=iteration_tracking
    )
    log_experiment_result(result=experiment_result)

    end_tracing()


if __name__ == "__main__":
    main()
