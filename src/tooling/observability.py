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
from tooling.model import Label


def log_artifacts(artifact_paths: Union[Path, Iterable[Path]]) -> None:
    if isinstance(artifact_paths, Iterable):
        for path in artifact_paths:
            mlflow.log_artifact(path)
    else:
        mlflow.log_artifact(artifact_paths)


def log_metric(name: str, value: float, step: int) -> None:
    mlflow.log_metric(name, value, step)


def log_metrics(
    base_name: str, labels: List[Label], values: List[float], step: int
) -> None:
    for label, value in zip(labels, values, strict=True):
        mlflow.log_metric(f"{label}_{base_name}", value, step=step)


def log_kfold_metric(name: str, values: List[float]) -> None:
    mlflow.log_metric(f"{name}_mean", statistics.mean(values))
    mlflow.log_metric(f"{name}_max", max(values))
    mlflow.log_metric(f"{name}_min", min(values))


def log_config(cfg: DictConfig) -> None:
    df = pd.json_normalize(
        cast(Dict[str, str], OmegaConf.to_object(cfg)), sep="."
    )
    mlflow.log_params(params=df.to_dict(orient="records")[0])


def config_mlflow(cfg: DictConfig) -> str:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)
    mlflow.autolog()
    log_config(cfg)

    run_name = mlflow.active_run().info.run_name
    if run_name is None:
        raise ValueError("No mlflow run_name")
    return str(run_name)


def end_tracing() -> None:
    mlflow.end_run()
