# %%
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from classifiers.sner import classify_sentences
from classifiers.sner import create_config_file
from classifiers.sner import create_solution
from classifiers.sner import create_train_file
from classifiers.sner import load_classification_result
from classifiers.sner import load_solution
from classifiers.sner import sentences_to_token_df
from classifiers.sner import train_sner
from data import return_dataset
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tooling import evaluation
from tooling.loading import import_dataset
from tooling.loading import load_dataset
from tooling.model import TORE_LABELS
from tooling.observability import config_mlflow
from tooling.observability import end_tracing
from tooling.observability import log_artifacts
from tooling.observability import log_metric
from tooling.sampling import LABELS_TEST
from tooling.sampling import LABELS_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset
from tooling.sampling import TEXT_TEST
from tooling.transformation import transform_dataset


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # %%
    print(OmegaConf.to_yaml(cfg))

    run_name = config_mlflow(cfg)

    # %%

    dataset = return_dataset(cfg.dataset.name)
    dataset_path = import_dataset(name=run_name, ds_spec=dataset)
    log_artifacts(dataset_path)
    # %%
    d = load_dataset(name=run_name)

    d = transform_dataset(d)

    split_dataset_paths = split_dataset(
        name=run_name,
        text=d["text"],
        labels=d["self"],
        test_size=cfg.experiment.test_size,
        stratify=d["source"],
        random_state=cfg.experiment.random_state,
    )
    log_artifacts(split_dataset_paths)
    # %%
    labels_train = load_split_dataset(name=run_name, filename=LABELS_TRAIN)

    create_train_file(name=run_name, sentences=labels_train)
    train_paths = create_config_file(name=run_name)
    train_sner(name=run_name)

    log_artifacts(train_paths)
    # %%

    text_test = load_split_dataset(name=run_name, filename=TEXT_TEST)
    classification_result_path = classify_sentences(
        name=run_name, sentences=text_test
    )

    log_artifacts(classification_result_path)

    # %%

    labels_test = load_split_dataset(name=run_name, filename=LABELS_TEST)
    solution_path = create_solution(name=run_name, labels_test=labels_test)
    log_artifacts(solution_path)

    # %%
    results = load_classification_result(name=run_name)
    solution = load_solution(name=run_name)

    p = evaluation.score_precision(
        solution=solution["label"],
        results=results["label"],
        labels=TORE_LABELS,
    )
    log_metric("precision", p)

    r = evaluation.score_recall(
        solution=solution["label"],
        results=results["label"],
        labels=TORE_LABELS,
    )
    log_metric("recall", r)

    confusion_path = evaluation.confusion_matrix(
        name=run_name,
        solution=solution["label"],
        results=results["label"],
    )
    log_artifacts(confusion_path)

    end_tracing()


if __name__ == "__main__":
    main()
