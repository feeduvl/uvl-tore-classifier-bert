from pathlib import Path
from typing import List

import hydra
import mlflow
import pandas as pd
from classifiers.sner import classify_sentences
from classifiers.sner import create_config_file
from classifiers.sner import create_solution
from classifiers.sner import create_train_file
from classifiers.sner import load_classification_result
from classifiers.sner import load_solution
from classifiers.sner import train_sner
from data import return_dataset
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tooling import evaluation
from tooling.loading import import_dataset
from tooling.loading import load_dataset
from tooling.model import TORE_LABELS
from tooling.model import TORE_LABELS_0
from tooling.observability import config_mlflow
from tooling.observability import end_tracing
from tooling.observability import log_artifacts
from tooling.observability import log_kfold_metric
from tooling.observability import log_metric
from tooling.observability import log_metrics
from tooling.sampling import DATA_TEST
from tooling.sampling import DATA_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset_k_fold
from tooling.transformation import transform_dataset


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    run_name = config_mlflow(cfg)

    dataset = return_dataset(cfg.dataset.name)
    dataset_path = import_dataset(name=run_name, ds_spec=dataset)
    log_artifacts(dataset_path)

    d = load_dataset(name=run_name)

    transformed_d = transform_dataset(d)

    weighted_precision: List[float] = []
    weighted_recall: List[float] = []
    per_label_precision: List[float] = []
    per_label_recall: List[float] = []

    all_solutions = []
    all_results = []

    for iteration, dataset_paths in split_dataset_k_fold(
        name=run_name,
        dataset=transformed_d,
        folds=cfg.experiment.folds,
        random_state=cfg.experiment.random_state,
    ):
        log_artifacts(dataset_paths)

        data_train = load_split_dataset(
            name=run_name, filename=DATA_TRAIN, iteration=iteration
        )

        train_file = create_train_file(
            name=run_name, iteration=iteration, data_train=data_train
        )
        log_artifacts(train_file)

        config_path = create_config_file(name=run_name, iteration=iteration)
        log_artifacts(config_path)

        model_path = train_sner(name=run_name, iteration=iteration)
        log_artifacts(model_path)

        data_test = load_split_dataset(
            name=run_name, iteration=iteration, filename=DATA_TEST
        )
        classification_result_paths = classify_sentences(
            name=run_name, iteration=iteration, data_test=data_test
        )

        log_artifacts(classification_result_paths)

        solution_paths = create_solution(
            name=run_name, iteration=iteration, data_test=data_test
        )
        log_artifacts(solution_paths)

        results = load_classification_result(
            name=run_name, iteration=iteration
        )
        solution = load_solution(name=run_name, iteration=iteration)

        p = evaluation.score_precision(
            solution=solution["tore_label"],
            results=results["tore_label"],
            labels=TORE_LABELS,
            average=cfg.experiment.precision_average,
        )
        weighted_precision.append(p)
        print(f"Weighted precision: {p}")
        log_metric(name="weighted_precision", value=p, step=iteration)

        r = evaluation.score_recall(
            solution=solution["tore_label"],
            results=results["tore_label"],
            labels=TORE_LABELS_0,
            average=cfg.experiment.recall_average,
        )
        weighted_recall.append(r)
        print(f"Weighted recall: {r}")
        log_metric("weighted_recall", value=r, step=iteration)

        pl_p = evaluation.score_precision(
            solution=solution["tore_label"],
            results=results["tore_label"],
            labels=TORE_LABELS_0,
            average=None,
        )
        per_label_precision.append(pl_p)
        log_metrics(
            base_name="precision",
            labels=TORE_LABELS_0,
            values=pl_p,
            step=iteration,
        )

        pl_r = evaluation.score_recall(
            solution=solution["tore_label"],
            results=results["tore_label"],
            labels=TORE_LABELS_0,
            average=None,
        )
        per_label_recall.append(pl_r)
        log_metrics(
            base_name="recall",
            labels=TORE_LABELS_0,
            values=pl_r,
            step=iteration,
        )

        confusion_path = evaluation.confusion_matrix(
            name=run_name,
            iteration=iteration,
            solution=solution["tore_label"],
            results=results["tore_label"],
        )

        all_solutions.append(solution["tore_label"])
        all_results.append(results["tore_label"])

        log_artifacts(confusion_path)

    log_kfold_metric(name="weighted_precision", values=weighted_precision)
    log_kfold_metric(name="weighted_recall", values=weighted_recall)

    all_solutions_df = pd.concat(all_solutions)
    all_results_df = pd.concat(all_results)

    confusion_path = evaluation.sum_confusion_matrix(
        name=run_name, solution=all_solutions_df, results=all_results_df
    )
    log_artifacts(confusion_path)

    end_tracing()


if __name__ == "__main__":
    main()
