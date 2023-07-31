from typing import List

import mlflow

from classifiers.sner.classifier import classify_sentences
from classifiers.sner.classifier import create_config_file
from classifiers.sner.classifier import create_solution
from classifiers.sner.classifier import create_train_file
from classifiers.sner.classifier import train_sner
from classifiers.sner.files import load_result
from classifiers.sner.files import load_solution
from tooling import evaluation
from tooling.config import SNERConfig
from tooling.loading import import_dataset
from tooling.logging import logging_setup
from tooling.model import get_id2label
from tooling.model import get_label2id
from tooling.observability import end_tracing
from tooling.sampling import DATA_TEST
from tooling.sampling import DATA_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset_k_fold
from tooling.transformation import transform_dataset
from tooling.types import IterationResult


logging = logging_setup(__name__)


def sner_pipeline(cfg: SNERConfig, run_name: str) -> None:
    # Import Dataset
    import_dataset(cfg, run_name)

    # Transform Dataset
    transformed_dataset = transform_dataset(
        cfg, run_name, fill_with_zeros=False
    )
    id2label = get_id2label(transformed_dataset["labels"])
    label2id = get_label2id(transformed_dataset["labels"])
    mlflow.log_param("id2label", id2label)
    mlflow.log_param("label2id", label2id)

    # Prepare evaluation tracking
    iteration_tracking: List[IterationResult] = []

    # Start kfold
    for iteration, dataset_paths in split_dataset_k_fold(
        name=run_name,
        dataset=transformed_dataset["dataset"],
        cfg_experiment=cfg.experiment,
    ):
        logging.info(f"Starting {iteration=}")

        # Load training data
        data_train = load_split_dataset(
            name=run_name, filename=DATA_TRAIN, iteration=iteration
        )

        create_train_file(
            name=run_name, iteration=iteration, data_train=data_train
        )

        create_config_file(name=run_name, iteration=iteration)

        # Train
        train_sner(name=run_name, iteration=iteration)

        # Classify
        data_test = load_split_dataset(
            name=run_name, iteration=iteration, filename=DATA_TEST
        )

        classify_sentences(
            name=run_name, iteration=iteration, data_test=data_test
        )

        # Create Solution
        create_solution(
            name=run_name, iteration=iteration, data_test=data_test
        )

        # Evaluate Iteration
        evaluation.evaluate_iteration(
            run_name=run_name,
            iteration=iteration,
            average=cfg.experiment.average,
            labels=transformed_dataset["labels"],
            solution=load_solution(name=run_name, iteration=iteration),
            result=load_result(name=run_name, iteration=iteration),
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
