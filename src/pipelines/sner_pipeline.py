from typing import List

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from classifiers.sner import classify_sentences
from classifiers.sner import create_config_file
from classifiers.sner import create_solution
from classifiers.sner import create_train_file
from classifiers.sner import train_sner
from classifiers.sner.files import load_result
from classifiers.sner.files import load_solution
from data import get_dataset_information
from tooling import evaluation
from tooling.config import SNERConfig
from tooling.config import Transformation
from tooling.loading import import_dataset
from tooling.loading import load_dataset
from tooling.logging import logging_setup
from tooling.observability import config_mlflow
from tooling.observability import end_tracing
from tooling.observability import log_artifacts
from tooling.observability import log_experiment_result
from tooling.observability import log_iteration_result
from tooling.sampling import DATA_TEST
from tooling.sampling import DATA_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset_k_fold
from tooling.transformation import lower_case_token
from tooling.transformation import transform_dataset

cs = ConfigStore.instance()
cs.store(name="base_config", node=SNERConfig)
cs.store(
    group="transformation", name="base_label_activity", node=Transformation
)

logging = logging_setup()


@hydra.main(version_base=None, config_path="conf", config_name="config_sner")
def main(cfg: SNERConfig) -> None:
    # Setup experiment
    run_name = config_mlflow(cfg)

    # Import Dataset
    import_dataset(cfg, run_name)

    # Transform Dataset
    transformed_dataset = transform_dataset(
        cfg, run_name, fill_with_zeros=False
    )

    # Prepare evaluation tracking
    iteration_tracking: List[evaluation.IterationResult] = []

    # Start kfold
    for iteration, dataset_paths in split_dataset_k_fold(
        name=run_name,
        dataset=transformed_dataset["dataset"],
        folds=cfg.experiment.folds,
        random_state=cfg.experiment.random_state,
    ):
        log_artifacts(dataset_paths)

        # Load training data and create a training file
        data_train = load_split_dataset(
            name=run_name, filename=DATA_TRAIN, iteration=iteration
        )
        train_file = create_train_file(
            name=run_name, iteration=iteration, data_train=data_train
        )
        log_artifacts(train_file)

        # Configure Training
        config_path = create_config_file(name=run_name, iteration=iteration)
        log_artifacts(config_path)

        # Train
        model_path = train_sner(name=run_name, iteration=iteration)
        log_artifacts(model_path)

        # Classify
        data_test = load_split_dataset(
            name=run_name, iteration=iteration, filename=DATA_TEST
        )
        classification_result_paths = classify_sentences(
            name=run_name, iteration=iteration, data_test=data_test
        )
        log_artifacts(classification_result_paths)

        # Create Solution
        solution_paths = create_solution(
            name=run_name, iteration=iteration, data_test=data_test
        )
        log_artifacts(solution_paths)

        # Evaluate Iteration
        result = load_result(name=run_name, iteration=iteration)
        solution = load_solution(name=run_name, iteration=iteration)

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
