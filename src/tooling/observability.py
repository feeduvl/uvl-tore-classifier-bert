import collections
from collections.abc import (
    Iterable,
)
from collections.abc import Iterator
from collections.abc import MutableMapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import mlflow
from omegaconf import OmegaConf
from pandas.errors import UndefinedVariableError

from data.tooling import cleanup_files
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


def check_rerun(cfg: Config) -> None:
    mlflow.set_tracking_uri(cfg.meta.mlflow_tracking_uri)

    logging.info("\n" + OmegaConf.to_yaml(cfg))

    run_id = get_run_id(cfg)
    if run_id:
        if cfg.experiment.force:
            logging.warn("Experiment was already run, continuing")
        else:
            logging.warn("Experiment was already run, aborting")
            raise RerunException
    else:
        logging.info("New experiment. Running")


@contextmanager
def config_mlflow(cfg: Config) -> Iterator[mlflow.ActiveRun]:
    experiment = mlflow.get_experiment_by_name(cfg.experiment.name)

    nested = False
    if mlflow.active_run():
        nested = True

    with mlflow.start_run(
        experiment_id=experiment.experiment_id, nested=nested
    ) as current_run:
        mlflow.autolog(silent=True, log_models=False)
        log_config(cfg)
        yield current_run


class RerunException(Exception):
    pass


def get_run_id(
    cfg: Config,
) -> Optional[str]:
    experiment_id = mlflow.set_experiment(cfg.experiment.name)
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id.experiment_id], output_format="pandas"
    )
    runs_df.rename(columns=lambda x: x.replace(".", "_"), inplace=True)

    config = flatten(
        cast(Dict[str, str], OmegaConf.to_container(cfg, resolve=True)),
        separator="_",
    )
    del config["experiment_force"]

    params_query_string = " and ".join(
        [f"params_{key} == '{value}'" for key, value in config.items()]
    )

    mlflow_config: Dict[str, str] = {}
    mlflow_config[
        "tags_mlflow_source_git_commit"
    ] = mlflow.utils.git_utils.get_git_commit(".")
    mlflow_config["status"] = "FINISHED"

    mlflow_config_query_string = " and ".join(
        [f"{key} == '{value}'" for key, value in mlflow_config.items()]
    )

    query_string = params_query_string + " and " + mlflow_config_query_string

    try:
        res = runs_df.query(query_string)
    except (
        UndefinedVariableError
    ):  # there is parameter in the configuration that isn't known to mlflow. we can force a new run because at least the code has changed (even if the git hash hasn't)
        res = []

    if len(res) != 0:
        run_id = str(res["run_id"].iloc[-1])
        return run_id
    else:
        return None


def end_tracing() -> None:
    cleanup_files()


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
    if result.confusion_matrix:
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
