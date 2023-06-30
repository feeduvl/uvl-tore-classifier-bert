import statistics
from collections.abc import (
    Iterable,
)
from pathlib import Path
from typing import cast
from typing import Dict
from typing import List
from typing import Union

import mlflow
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf

from tooling.evaluation import ExperimentResult
from tooling.evaluation import IterationResult
from tooling.model import Label


def log_artifacts(artifact_paths: Union[Path, Iterable[Path]]) -> None:
    if isinstance(artifact_paths, Iterable):
        for path in artifact_paths:
            mlflow.log_artifact(path)
    else:
        mlflow.log_artifact(artifact_paths)


def log_config(cfg: DictConfig) -> None:
    df = pd.json_normalize(
        cast(Dict[str, str], OmegaConf.to_object(cfg)), sep="."
    )
    mlflow.log_params(params=df.to_dict(orient="records")[0])


def config_mlflow(cfg: DictConfig) -> str:
    mlflow.set_tracking_uri(cfg.meta.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)
    mlflow.autolog()
    log_config(cfg)

    run_name = mlflow.active_run().info.run_name
    if run_name is None:
        raise ValueError("No mlflow run_name")
    return str(run_name)


def end_tracing() -> None:
    mlflow.end_run()


def log_iteration_result(result: IterationResult) -> None:
    mlflow.log_metric("precision", result.precision, step=result.step)
    mlflow.log_metric("recall", result.recall, step=result.step)

    pl_precision = {
        f"precision_{label}": value
        for label, value in result.pl_precision.items()
    }
    mlflow.log_metrics(pl_precision)

    pl_recall = {
        f"recall_{label}": value for label, value in result.pl_recall.items()
    }
    mlflow.log_metrics(pl_recall)
    mlflow.log_artifact(result.confusion_matrix)
    return None


def log_experiment_result(result: ExperimentResult) -> None:
    mlflow.log_metric("min_precision", result.min_precision)
    mlflow.log_metric("max_precision", result.max_precision)
    mlflow.log_metric("mean_precision", result.mean_precision)

    mlflow.log_metric("min_recall", result.min_recall)
    mlflow.log_metric("max_recall", result.max_recall)
    mlflow.log_metric("mean_recall", result.mean_recall)

    pl_mean_precision = {
        f"mean_precision_{label}": value
        for label, value in result.pl_mean_precision.items()
    }
    mlflow.log_metrics(pl_mean_precision)

    pl_mean_recall = {
        f"mean_recall_{label}": value
        for label, value in result.pl_mean_recall.items()
    }
    mlflow.log_metrics(pl_mean_recall)

    mlflow.log_artifact(result.confusion_matrix)

    return None
