import collections
from collections.abc import (
    Iterable,
)
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import mlflow
from omegaconf import OmegaConf

from tooling.config import Config
from tooling.logging import logging_setup
from tooling.types import ExperimentResult
from tooling.types import IterationResult


logging = logging_setup(__name__)


def log_artifacts(
    artifact_paths: Union[Path, Iterable[Path], Dict[str, Any]]
) -> None:
    if isinstance(artifact_paths, Dict):
        for value in artifact_paths.values():
            if isinstance(value, Path):
                mlflow.log_artifact(value)
    elif isinstance(artifact_paths, Iterable):
        for path in artifact_paths:
            mlflow.log_artifact(path)
    else:
        mlflow.log_artifact(artifact_paths)


def flatten(
    dictionary: MutableMapping[str, Any],
    parent_key: Optional[str] = None,
    separator: str = "_",
) -> Dict[str, Any]:
    """
    Turn a nested dictionary into a flattened dictionary
    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary
    """

    items: List[Tuple[str, Any]] = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, collections.abc.MutableMapping):
            items.extend(flatten(value, new_key, separator).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def log_config(cfg: Config) -> None:
    config = flatten(
        cast(Dict[str, str], OmegaConf.to_container(cfg, resolve=True)),
        separator=".",
    )

    mlflow.log_params(config)


def config_mlflow(cfg: Config) -> str:
    mlflow.set_tracking_uri(cfg.meta.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment.name)
    mlflow.autolog(silent=True)
    log_config(cfg)
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    run_name = mlflow.active_run().info.run_name
    if run_name is None:
        raise ValueError("No mlflow run_name")
    return str(run_name)


def end_tracing() -> None:
    mlflow.end_run()


def log_iteration_result(result: IterationResult) -> None:
    mlflow.log_metric(
        "iteration_label_count", result.label_count, step=result.step
    )

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
    mlflow.log_metric("label_count", result.label_count)

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
