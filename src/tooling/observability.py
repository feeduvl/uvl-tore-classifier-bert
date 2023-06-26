from collections.abc import (
    Iterable,
)
from pathlib import Path
from typing import Union

import mlflow
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf


def log_artifacts(artifact_paths: Union[Path, Iterable[Path]]):
    if isinstance(artifact_paths, Iterable):
        for path in artifact_paths:
            mlflow.log_artifact(path)
    else:
        mlflow.log_artifact(artifact_paths)


def log_metric(name: str, value: any):
    mlflow.log_metric(name, value)


def log_config(cfg: DictConfig):
    df = pd.json_normalize(OmegaConf.to_container(cfg), sep=".")
    mlflow.log_params(params=df.to_dict(orient="records")[0])


def config_mlflow(cfg: DictConfig) -> str:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)
    mlflow.autolog()
    log_config(cfg)

    return mlflow.active_run().info.run_name


def end_tracing():
    mlflow.end_run()
