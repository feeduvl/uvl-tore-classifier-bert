import sys
from typing import List

import hydra
from hydra.core.config_store import ConfigStore

from classifiers.sner.classifier import classify_sentences
from classifiers.sner.classifier import create_config_file
from classifiers.sner.classifier import create_solution
from classifiers.sner.classifier import create_train_file
from classifiers.sner.classifier import train_sner
from classifiers.sner.files import load_result
from classifiers.sner.files import load_solution
from tooling import evaluation
from tooling.config import SNERConfig
from tooling.config import Transformation
from tooling.loading import import_dataset
from tooling.logging import logging_setup
from tooling.observability import config_mlflow
from tooling.observability import end_tracing
from tooling.observability import RerunException
from tooling.sampling import DATA_TEST
from tooling.sampling import DATA_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset_k_fold
from tooling.transformation import transform_dataset
from tooling.types import IterationResult

cs = ConfigStore.instance()
cs.store(name="base_config", node=SNERConfig)
cs.store(
    group="transformation", name="base_label_activity", node=Transformation
)

logging = logging_setup(__name__)


def main(cfg: SNERConfig) -> None:
    # Setup experiment
    try:
        run_name = config_mlflow(cfg)
    except RerunException:
        return

    # Import Dataset
    import_dataset(cfg, run_name)

    # Transform Dataset
    transformed_dataset = transform_dataset(
        cfg, run_name, fill_with_zeros=False
    )

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

    end_tracing(status="FINISHED")


@hydra.main(version_base=None, config_path="conf", config_name="config_sner")
def main_wrapper(cfg: SNERConfig) -> None:
    try:
        main(cfg)

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


if __name__ == "__main__":
    main_wrapper
