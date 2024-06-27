import sys

import hydra
from hydra.core.config_store import ConfigStore

from classifiers.roberta.pipeline import roberta_pipeline
from tooling.config import RoBERTaConfig
from tooling.logging import logging_setup
from tooling.observability import check_rerun
from tooling.observability import config_mlflow
from tooling.observability import end_tracing
from tooling.observability import RerunException

cs = ConfigStore.instance()
cs.store(name="base_config", node=RoBERTaConfig)

logging = logging_setup(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config_bert")
def roberta(cfg: RoBERTaConfig) -> None:
    try:
        check_rerun(cfg=cfg)
    except RerunException:
        return

    logging.info("Entering mlflow context")
    with config_mlflow(cfg=cfg) as current_run:
        try:
            roberta_pipeline(cfg, run_name=current_run.info.run_name)
            end_tracing()

        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received")
            end_tracing()
            sys.exit()

        except Exception as e:
            logging.error(e)
            end_tracing()
            raise e

    logging.info("Left mlflow context")
    return


if __name__ == "__main__":
    roberta()
