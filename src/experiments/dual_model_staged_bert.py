import sys

import hydra
from hydra.core.config_store import ConfigStore

from classifiers.staged_bert.pipeline import dual_stage_bert_pipeline
from tooling.config import DualModelStagedBERTConfig
from tooling.logging import logging_setup
from tooling.observability import check_rerun
from tooling.observability import config_mlflow
from tooling.observability import end_tracing
from tooling.observability import RerunException

logging = logging_setup(__name__)

cs = ConfigStore.instance()
cs.store(name="base_config", node=DualModelStagedBERTConfig)


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="2_stage_dual_model_bert_bert_sweep",
)
def dual_stage_bert(cfg: DualModelStagedBERTConfig) -> None:
    try:
        check_rerun(cfg=cfg)
    except RerunException:
        return

    logging.info("Entering mlflow context")
    with config_mlflow(cfg=cfg) as current_run:
        try:
            dual_stage_bert_pipeline(cfg, run_name=current_run.info.run_name)
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
    dual_stage_bert()
