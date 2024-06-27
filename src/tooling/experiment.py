import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from typing import Dict
from typing import Optional
from typing import Tuple

import mlflow
import pandas as pd
import tensorflow as tf
from datasets import Dataset
from strictly_typed_pandas.dataset import DataSet
from transformers import BertForTokenClassification
from transformers import BertTokenizerFast
from transformers import Trainer

from classifiers.bert.classifier import create_bert_dataset
from classifiers.bert.classifier import create_staged_bert_dataset
from classifiers.bilstm import get_glove_model
from classifiers.bilstm.classifier import classify_with_bilstm
from classifiers.sner.classifier import classify_sentences_action
from classifiers.sner.classifier import classify_with_sner
from classifiers.sner.classifier import realign_results
from classifiers.staged_bert.classifier import classify_with_bert
from data import staged_bert_filepath
from experiments.bert import bert
from experiments.bilstm import bilstm
from experiments.sner import sner
from tooling.config import DualModelStagedBERTConfig
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
    cfg: DualModelStagedBERTConfig,
    run_name: str,
    iteration: int,
    retry: int = 0,
) -> Tuple[
    Dict[Label_None_Pad, int],
    Dict[int, Label_None_Pad],
    Optional[pd.DataFrame],
    Optional[Sequence[Label_None_Pad]],
]:
    if retry > 1:
        raise RecursionError

    if cfg.first_model_bert:
        run_id = get_run_id(cfg.first_model_bert)
        suffix = f"/{iteration}_model"
    elif cfg.first_model_bilstm:
        run_id = get_run_id(cfg.first_model_bilstm)
        suffix = f"/{iteration}_model"
    elif cfg.first_model_sner:
        run_id = get_run_id(cfg.first_model_sner)
        suffix = f"/{iteration}_model.ser.gz"
    else:
        raise NotImplementedError

    if run_id:
        logging.info(
            f"Found existing run with run_id: {run_id} matching the configuration"
        )

        run = mlflow.get_run(run_id=run_id)
        artifact_uri = run.info.artifact_uri
        path_pretrained_model = pretrained_model_path(name=run_name)
        artifact_path = artifact_uri + suffix
        logging.info(f"Downloading run model from {artifact_path}")
        mlflow.artifacts.download_artifacts(
            artifact_path, dst_path=path_pretrained_model
        )
        hint_label2id = json.loads(
            run.data.params["label2id"].replace("'", '"')
        )
        hint_id2label = {v: k for k, v in hint_label2id.items()}

        glove_model = None
        padded_labels = None
        if cfg.first_model_bilstm:
            glove_model = get_glove_model()
            padded_labels = json.loads(
                run.data.params["padded_labels"].replace("'", '"')
            )

        return (
            cast(Dict[Label_None_Pad, int], hint_label2id),
            cast(Dict[int, Label_None_Pad], hint_id2label),
            glove_model,
            padded_labels,
        )

    else:
        if cfg.first_model_bert:
            logging.warn("Running missing experiment")
            bert(cfg.first_model_bert)
            logging.warn("Retry finding experiment")
            return get_model(
                cfg, run_name=run_name, iteration=iteration, retry=retry + 1
            )
        if cfg.first_model_bilstm:
            logging.warn("Running missing experiment")
            bilstm(cfg.first_model_bilstm)
            logging.warn("Retry finding experiment")
            return get_model(
                cfg, run_name=run_name, iteration=iteration, retry=retry + 1
            )
        if cfg.first_model_sner:
            logging.warn("Running missing experiment")
            sner(cfg.first_model_sner)
            logging.warn("Retry finding experiment")
            return get_model(
                cfg, run_name=run_name, iteration=iteration, retry=retry + 1
            )
        else:
            raise NotImplementedError


def run_model(
    cfg: DualModelStagedBERTConfig,
    run_name: str,
    iteration: int,
    data: DataSet[DataDF],
    label2id: Dict[Label_None_Pad, int],
    tokenizer: BertTokenizerFast,
    max_len: int,
    glove_model: Optional[pd.DataFrame],
    padded_labels: Optional[Sequence[Label_None_Pad]],
    hint_label2id: Optional[Dict[Label_None_Pad, int]] = None,
    hint_id2label: Optional[Dict[int, Label_None_Pad]] = None,
) -> Optional[Dataset]:
    if not hint_label2id:
        raise ValueError("No 'hint_label2id' provided")
    if not hint_id2label:
        raise ValueError("No 'hint_id2label' provided")
    if not label2id:
        raise ValueError("No 'label2id' provided")

    if cfg.first_model_bert:
        model_path = pretrained_model_path(name=run_name).joinpath(
            Path(f"{iteration}_model")
        )
        first_bert_data = create_bert_dataset(
            input_data=data,
            label2id=label2id,  # we have all labels in the data, therefore provide label2id - classify with bert won't use them
            tokenizer=tokenizer,
            max_len=max_len,
        )

        hints = classify_with_bert(
            model_path=model_path,
            bert_data=first_bert_data,
            id2label=hint_id2label,
        )

        hint_ids = [
            [hint_label2id[hint] for hint in hint_list] for hint_list in hints
        ]
        second_bert_data = create_bert_dataset(
            input_data=data,
            label2id=label2id,
            tokenizer=tokenizer,
            max_len=max_len,
        )

        return second_bert_data.add_column(
            name="hint_input_ids", column=hint_ids
        )

    if cfg.first_model_sner:
        model_path = pretrained_model_path(name=run_name).joinpath(
            Path(f"{iteration}_model.ser.gz")
        )

        hinted_data = classify_with_sner(model_path=model_path, data=data)

        bert_data = create_staged_bert_dataset(
            input_data=hinted_data,
            label2id=label2id,
            hint_label2id=hint_label2id,
            tokenizer=tokenizer,
            max_len=max_len,
        )

        return bert_data

    if cfg.first_model_bilstm:
        if not hint_id2label:
            raise ValueError("No 'hint_id2label' provided")
        if not glove_model:
            raise ValueError("No 'glove_model' provided")
        if not cfg.first_model_bilstm.bilstm.sentence_length:
            raise ValueError("No 'sentence_length' provided")

        model_path = pretrained_model_path(name=run_name).joinpath(
            Path(f"{iteration}_model")
        )

        hinted_data = classify_with_bilstm(
            model_path=model_path,
            data=data,
            max_len=cfg.first_model_bilstm.bilstm.sentence_length,
            glove_model=glove_model,
            hint_id2label=hint_id2label,
        )

        bert_data = create_staged_bert_dataset(
            input_data=hinted_data,
            label2id=label2id,
            hint_label2id=hint_label2id,
            tokenizer=tokenizer,
            max_len=max_len,
        )

        return bert_data

    raise NotImplementedError
