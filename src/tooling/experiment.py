from pathlib import Path
from typing import Dict
from typing import Optional

import mlflow
from datasets import Dataset
from strictly_typed_pandas.dataset import DataSet
from transformers import BertForTokenClassification
from transformers import BertTokenizerFast
from transformers import Trainer

from classifiers.bert.classifier import create_bert_dataset
from classifiers.bert.classifier import setup_device
from classifiers.staged_bert.classifier import get_hint_column
from data import staged_bert_filepath
from pipelines.bert_pipeline import bert
from tooling.config import BERTConfig
from tooling.config import BiLSTMConfig
from tooling.config import DualModelStagedBERTConfig
from tooling.config import FirstStageConfigs
from tooling.config import SNERConfig
from tooling.config import StagedBERTConfig
from tooling.logging import logging_setup
from tooling.model import DataDF
from tooling.model import Label_None_Pad
from tooling.observability import get_run_id


DOWNLOAD_MODEL_FILENAME = "pretrained"

logging = logging_setup(__name__)


def pretrained_model_path(name: str) -> Path:
    return staged_bert_filepath(
        name=name,
        filename=DOWNLOAD_MODEL_FILENAME,
    )


def get_model(
    cfg: DualModelStagedBERTConfig, run_name: str, retry: int = 0
) -> None:
    if retry > 1:
        raise RecursionError

    if cfg.first_model_bert:
        run_id = get_run_id(cfg.first_model_bert)
    else:
        raise NotImplementedError

    if run_id:
        logging.info(
            f"Found existing run with run_id: {run_id} matching the configuration"
        )

        run = mlflow.get_run(run_id=run_id)
        artifact_uri = run.info.artifact_uri
        path_pretrained_model = pretrained_model_path(name=run_name)
        artifact_path = artifact_uri + "/0_model"
        logging.info(f"Downloading run model from {artifact_path}")
        mlflow.artifacts.download_artifacts(
            artifact_path, dst_path=path_pretrained_model
        )

    else:
        if cfg.first_model_bert:
            logging.warn("Running missing experiment")
            bert(cfg.first_model_bert)
            logging.warn("Retry finding experiment")
            return get_model(cfg, run_name=run_name, retry=retry + 1)
        else:
            raise NotImplementedError


def run_model(
    cfg: DualModelStagedBERTConfig,
    run_name: str,
    data: DataSet[DataDF],
    label2id: Dict[Label_None_Pad, int],
    tokenizer: BertTokenizerFast,
    max_len: int,
) -> Optional[Dataset]:
    if cfg.first_model_bert:
        logging.info("Converting dataframe to bert dataset")
        bert_data = create_bert_dataset(
            input_data=data,
            label2id=label2id,
            tokenizer=tokenizer,
            max_len=max_len,
        )

        if not bert_data:
            raise ValueError("No BERT Dataset supplied")

        logging.info("Loading Model")
        model = BertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path(
                name=run_name
            ).joinpath(Path("0_model"))
        )
        model.to(device=setup_device())
        trainer = Trainer(model=model)

        logging.info("Creating hint column")
        # Add predicted hint labels to train_dataset
        first_stage_train_result = trainer.predict(
            bert_data.remove_columns("labels")
        )
        hint_column = get_hint_column(first_stage_train_result[0])
        bert_data = bert_data.add_column("hint_input_ids", hint_column)
        return bert_data

    raise NotImplementedError
