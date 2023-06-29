from typing import cast
from typing import List

import hydra
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf
from strictly_typed_pandas import DataSet

from classifiers.bilstm import get_glove_model
from classifiers.bilstm import get_one_hot_encoding
from classifiers.bilstm import get_word_embeddings
from data import get_dataset_information
from tooling import evaluation
from tooling.loading import import_dataset
from tooling.loading import load_dataset
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Setup experiment
    print(OmegaConf.to_yaml(cfg))
    run_name = config_mlflow(cfg)

    # Download Model
    glove_model = get_glove_model()

    # Import Dataset
    dataset_information = get_dataset_information(cfg.dataset.name)
    imported_dataset_path = import_dataset(
        name=run_name, ds_spec=dataset_information
    )
    log_artifacts(imported_dataset_path)

    # Transform Dataset
    d = load_dataset(name=run_name)
    dataset_labels, transformed_d = transform_dataset(
        dataset=d, cfg=cfg.transformation
    )

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
            dataset=data_train, sentence_length=None
        )
        embeddings = get_word_embeddings(
            dataset=data_train, glove_model=glove_model, sentence_length=None
        )

        # Configure Training

        # Train

        # Classify

        # Create Solution

        # Evaluate Iteration

    # Evaluate Run
    experiment_result = evaluation.evaluate_experiment(
        run_name=run_name, iteration_results=iteration_tracking
    )
    log_experiment_result(result=experiment_result)

    end_tracing()


if __name__ == "__main__":
    main()
