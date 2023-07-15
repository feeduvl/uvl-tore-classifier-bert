from collections.abc import Sequence
from typing import cast
from typing import List

import hydra
import numpy as np
import tensorflow as tf
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from rich.logging import RichHandler
from strictly_typed_pandas import DataSet
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import BertConfig
from transformers import BertForTokenClassification
from transformers import BertTokenizer
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments
from transformers.utils import logging

from classifiers.bert.classifier import create_tore_dataset
from classifiers.bert.classifier import get_compute_metrics
from classifiers.bert.classifier import get_tokenize_and_align_labels
from classifiers.bert.classifier import prepare_data
from classifiers.bert.classifier import tokenize_and_align_labels
from classifiers.bert.files import model_path
from classifiers.bert.files import output_path
from data import get_dataset_information
from tooling import evaluation
from tooling.config import BiLSTMConfig
from tooling.loading import import_dataset
from tooling.loading import load_dataset
from tooling.model import data_to_list_of_token_lists
from tooling.model import get_id2label
from tooling.model import get_label2id
from tooling.model import get_labels
from tooling.model import Label_None_Pad
from tooling.model import PAD
from tooling.model import ResultDF
from tooling.model import ZERO
from tooling.observability import config_mlflow
from tooling.observability import end_tracing
from tooling.observability import log_artifacts
from tooling.observability import log_experiment_result
from tooling.observability import log_iteration_result
from tooling.sampling import DATA_TEST
from tooling.sampling import DATA_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset_k_fold
from tooling.transformation import transform_dataset


cs = ConfigStore.instance()
cs.store(name="base_config", node=BiLSTMConfig)

MAX_LEN = 128
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
MAX_GRAD_NORM = 10
TOKENIZER = BertTokenizerFast.from_pretrained("bert-base-uncased")


@hydra.main(version_base=None, config_path="conf", config_name="config_bert")
def main(cfg: BiLSTMConfig) -> None:
    # Setup experiment
    print(OmegaConf.to_yaml(cfg))
    run_name = config_mlflow(cfg)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Import Dataset
    dataset_information = get_dataset_information(cfg.experiment.dataset)
    imported_dataset_path = import_dataset(
        name=run_name, ds_spec=dataset_information
    )
    log_artifacts(imported_dataset_path)

    # Transform Dataset
    d = load_dataset(name=run_name)
    dataset_labels, transformed_d = transform_dataset(
        dataset=d, cfg=cfg.transformation
    )

    transformed_d.fillna(ZERO, inplace=True)

    sentence_length = cfg.bilstm.sentence_length
    if sentence_length is None:
        sentences_token_list = data_to_list_of_token_lists(data=transformed_d)
        sentence_length = max(
            [len(sentence_tl) for sentence_tl in sentences_token_list]
        )

    iteration_tracking: List[evaluation.IterationResult] = []

    # Start kfold
    for iteration, dataset_paths in split_dataset_k_fold(
        name=run_name,
        dataset=transformed_d,
        folds=cfg.experiment.folds,
        random_state=cfg.experiment.random_state,
    ):
        log_artifacts(dataset_paths)

        t_a_a_l = get_tokenize_and_align_labels(
            max_len=MAX_LEN, tokenizer=TOKENIZER
        )

        # Load training data and create a training file
        data_train = load_split_dataset(
            name=run_name, filename=DATA_TRAIN, iteration=iteration
        )
        train_data = create_tore_dataset(data=data_train)
        train_data = train_data.map(t_a_a_l, batched=True)
        train_data = train_data.rename_columns(
            {"string": "text", "tore_label_id": "labels"}
        )

        # Create Solution
        data_test = load_split_dataset(
            name=run_name, filename=DATA_TEST, iteration=iteration
        )

        test_data = create_tore_dataset(data=data_test)
        test_data = test_data.map(t_a_a_l, batched=True)
        test_data = test_data.rename_columns(
            {"string": "text", "tore_label_id": "labels"}
        )

        # Get Model

        model = BertForTokenClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(dataset_labels),
            id2label=get_id2label(dataset_labels),
            label2id=get_label2id(dataset_labels),
        )
        model.to(device)

        # Train
        data_collator = DataCollatorForTokenClassification(tokenizer=TOKENIZER)

        compute_metrics = get_compute_metrics(
            iteration_tracking=iteration_tracking,
            average=cfg.experiment.average,
            run_name=run_name,
        )

        model_dir = str(
            model_path(name=run_name, iteration=iteration)
        )  # pin iteration to avoid relogging parameter

        output_dir = str(
            output_path(name=run_name, iteration=iteration)
        )  # pin iteration to avoid relogging parameter

        training_args = TrainingArguments(
            output_dir=output_dir,
            run_name=run_name,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=1,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            optim="adamw_torch",
            push_to_hub=False,
            use_mps_device=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=TOKENIZER,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.save_model(output_dir=model_dir)

        model = BertForTokenClassification.from_pretrained(
            model_dir,
            num_labels=len(dataset_labels),
            id2label=get_id2label(dataset_labels),
            label2id=get_label2id(dataset_labels),
        )

        test_args = TrainingArguments(
            output_dir=output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=16,
            dataloader_drop_last=False,
        )

        trainer = Trainer(
            model=model,
            args=test_args,
            compute_metrics=compute_metrics,
            tokenizer=TOKENIZER,
        )

        test_results = trainer.predict(test_data)

        print(test_results)

    # Evaluate Run

    experiment_result = evaluation.evaluate_experiment(
        run_name=run_name, iteration_results=iteration_tracking
    )
    log_experiment_result(result=experiment_result)

    end_tracing()


if __name__ == "__main__":
    main()
