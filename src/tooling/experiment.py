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
from classifiers.bert.classifier import setup_device
from classifiers.bilstm import get_glove_model
from classifiers.bilstm.classifier import get_word_embeddings
from classifiers.bilstm.classifier import MultiClassPrecision
from classifiers.bilstm.classifier import MultiClassRecall
from classifiers.bilstm.classifier import reverse_one_hot_encoding
from classifiers.sner.classifier import classify_sentences_action
from classifiers.sner.classifier import realign_results
from classifiers.staged_bert.classifier import get_hint_column
from data import staged_bert_filepath
from pipelines.bert_pipeline import bert
from pipelines.bilstm_pipeline import bilstm
from pipelines.sner_pipeline import sner
from tooling.config import DualModelStagedBERTConfig
from tooling.logging import logging_setup
from tooling.model import DataDF
from tooling.model import get_sentence_lengths
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
        suffix = "/0_model"
    elif cfg.first_model_bilstm:
        run_id = get_run_id(cfg.first_model_bilstm)
        suffix = "/0_model"
    elif cfg.first_model_sner:
        run_id = get_run_id(cfg.first_model_sner)
        suffix = "/0_model.ser.gz"
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
            return get_model(cfg, run_name=run_name, retry=retry + 1)
        if cfg.first_model_bilstm:
            logging.warn("Running missing experiment")
            bilstm(cfg.first_model_bilstm)
            logging.warn("Retry finding experiment")
            return get_model(cfg, run_name=run_name, retry=retry + 1)
        if cfg.first_model_sner:
            logging.warn("Running missing experiment")
            sner(cfg.first_model_sner)
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
    glove_model: Optional[pd.DataFrame],
    padded_labels: Optional[Sequence[Label_None_Pad]],
    hint_label2id: Optional[Dict[Label_None_Pad, int]] = None,
    hint_id2label: Optional[Dict[int, Label_None_Pad]] = None,
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

    if cfg.first_model_sner:
        if not hint_label2id:
            raise ValueError("No 'hint_label2id' provided")

        result = classify_sentences_action(
            modelfile=pretrained_model_path(name=run_name).joinpath(
                Path("0_model.ser.gz")
            ),
            data_test=data,
        )

        result = realign_results(input=data, output=result)

        data["hint_input_ids"] = result["tore_label"]

        bert_data = create_staged_bert_dataset(
            input_data=data,
            label2id=label2id,
            hint_label2id=hint_label2id,
            tokenizer=tokenizer,
            max_len=max_len,
        )

        return bert_data

    if cfg.first_model_bilstm:
        if not hint_label2id:
            raise ValueError("No 'hint_label2id' provided")
        if not hint_id2label:
            raise ValueError("No 'hint_id2label' provided")
        if not glove_model:
            raise ValueError("No 'glove_model' provided")
        if not padded_labels:
            raise ValueError("No 'padded_labels' provided")

        embeddings = get_word_embeddings(
            dataset=data,
            glove_model=glove_model,
            sentence_length=max_len,
        )

        trained_model = tf.keras.models.load_model(
            pretrained_model_path(name=run_name).joinpath(Path("0_model")),
            compile=False,  # https://github.com/tensorflow/tensorflow/issues/31850#issuecomment-578566637
            custom_objects={
                "MultiClassPrecision": MultiClassPrecision,
                "MultiClassRecall": MultiClassRecall,
            },
        )

        categorical_predictions = trained_model.predict(embeddings)

        label_df = reverse_one_hot_encoding(
            categorical_data=categorical_predictions,
            sentence_lengths=get_sentence_lengths(data),
            id2label=hint_id2label,
        )

        data.reset_index(drop=True, inplace=True)
        label_df.reset_index(drop=True, inplace=True)

        data["hint_input_ids"] = label_df["tore_label"]

        bert_data = create_staged_bert_dataset(
            input_data=data,
            label2id=label2id,
            hint_label2id=hint_label2id,
            tokenizer=tokenizer,
            max_len=max_len,
        )

        return bert_data

    raise NotImplementedError
