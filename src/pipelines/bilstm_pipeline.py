import sys
from collections.abc import Sequence
from typing import cast
from typing import List

import hydra
import numpy as np
import tensorflow as tf
from hydra.core.config_store import ConfigStore
from strictly_typed_pandas.dataset import DataSet

from classifiers.bilstm import get_embeddings_and_categorical
from classifiers.bilstm import get_glove_model
from classifiers.bilstm import get_model
from classifiers.bilstm import get_sentence_length
from classifiers.bilstm import reverse_one_hot_encoding
from classifiers.bilstm.files import model_path
from tooling import evaluation
from tooling.config import BiLSTMConfig
from tooling.loading import import_dataset
from tooling.logging import logging_setup
from tooling.model import Label_None_Pad
from tooling.model import PAD
from tooling.model import ResultDF
from tooling.observability import config_mlflow
from tooling.observability import end_tracing
from tooling.observability import RerunException
from tooling.sampling import DATA_TEST
from tooling.sampling import DATA_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset_k_fold
from tooling.transformation import get_class_weights
from tooling.transformation import transform_dataset
from tooling.types import IterationResult

# hide GPU because it is a lot slower than running without it
tf.config.set_visible_devices([], "GPU")

logging = logging_setup(__name__)

cs = ConfigStore.instance()
cs.store(name="base_config", node=BiLSTMConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config_bilstm")
def main(cfg: BiLSTMConfig) -> None:
    # Setup experiment
    run_name = config_mlflow(cfg)

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

    padded_labels: Sequence[Label_None_Pad] = transformed_dataset["labels"] + [
        PAD
    ]

    weights = cast(
        List[np.float32],
        get_class_weights(cfg.bilstm, transformed_dataset["dataset"]).tolist(),
    )
    weights.append(np.float32(1.0))  # account for padding label
    class_weights = {idx: value for idx, value in enumerate(weights)}

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
        )

        data_test = load_split_dataset(
            name=run_name, filename=DATA_TEST, iteration=iteration
        )

        processed_data_test = get_embeddings_and_categorical(
            dataset=data_test,
            sentence_length=sentence_length,
            labels=padded_labels,
            glove_model=glove_model,
        )

        # Get Model
        model = get_model(
            n_tags=len(padded_labels),
            sentence_length=sentence_length,
            cfg_bilstm=cfg.bilstm,
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

        # Classify

        trained_model = tf.keras.models.load_model(
            model_path(name=run_name, iteration=iteration),
            compile=False,  # https://github.com/tensorflow/tensorflow/issues/31850#issuecomment-578566637
        )

        categorical_predictions = trained_model.predict(
            processed_data_test["embeddings"]
        )

        # Create Solution
        solution = cast(DataSet[ResultDF], data_test[["string", "tore_label"]])

        # Evaluate Iteration
        result = reverse_one_hot_encoding(
            dataset=data_test,
            categorical_data=categorical_predictions,
            labels=padded_labels,
        )

        evaluation.evaluate_iteration(
            run_name=run_name,
            iteration=iteration,
            average=cfg.experiment.average,
            solution=solution,
            result=result,
            iteration_tracking=iteration_tracking,
        )

        logging.info(f"Finished {iteration=}")

        # early break if configured
        if iteration == cfg.experiment.iterations:
            logging.info(
                f"Breaking early after {iteration=} of {cfg.experiment.folds} folds"
            )
            break

    # Evaluate Run
    evaluation.evaluate_experiment(
        run_name=run_name, iteration_results=iteration_tracking
    )

    end_tracing(status="FINISHED")


if __name__ == "__main__":
    try:
        main()

    except RerunException:
        pass
    except KeyboardInterrupt:
        logging.info("Keyobard interrupt recieved")
        status = "FAILED"
        end_tracing(status=status)

        sys.exit()

    except Exception as e:
        logging.error(e)
        status = "FAILED"
        end_tracing(status=status)

        raise e
