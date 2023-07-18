from typing import List

import hydra
import mlflow
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
from transformers import TrainingArguments

from classifiers.bert.classifier import create_tore_dataset
from classifiers.bert.classifier import get_compute_metrics
from classifiers.bert.classifier import get_max_len
from classifiers.bert.classifier import get_tokenize_and_align_labels
from classifiers.bert.classifier import setup_device
from classifiers.bert.classifier import WeightedTrainer
from classifiers.bert.files import model_path
from classifiers.bert.files import output_path
from classifiers.staged_bert.classifier import generate_hint_data
from classifiers.staged_bert.classifier import get_hint_transformation
from classifiers.staged_bert.model import (
    StagedBertForTokenClassification,
)
from data import get_dataset_information
from tooling import evaluation
from tooling.config import StagedBERTConfig
from tooling.loading import import_dataset
from tooling.loading import load_dataset
from tooling.logging import logging_setup
from tooling.model import get_id2label
from tooling.model import get_label2id
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
from tooling.transformation import get_class_weights
from tooling.transformation import lower_case_token
from tooling.transformation import transform_dataset
from tooling.transformation import transform_token_label

cs = ConfigStore.instance()
cs.store(name="base_config", node=StagedBERTConfig)

logging = logging_setup()


@hydra.main(
    version_base=None, config_path="conf", config_name="config_staged_bert"
)
def main(cfg: StagedBERTConfig) -> None:
    run_name = config_mlflow(cfg)
    device = setup_device()

    # Import Dataset
    import_dataset(cfg, run_name)

    # Setup Tokenizer
    TOKENIZER = BertTokenizerFast.from_pretrained(cfg.bert.model)
    TOKENIZER.model_input_names.append("hint_input_ids")

    # Transform Dataset
    transformed_dataset = transform_dataset(
        cfg, run_name, fill_with_zeros=True
    )

    # Get model constants
    class_weights = get_class_weights(
        bert_cfg=cfg.bert, data=transformed_dataset["dataset"], device=device
    )
    id2label = get_id2label(transformed_dataset["labels"])
    label2id = get_label2id(transformed_dataset["labels"])

    max_len = get_max_len(
        bert_cfg=cfg.bert,
        data=transformed_dataset["dataset"],
        label2id=label2id,
        tokenizer=TOKENIZER,
    )

    hints = get_hint_transformation(cfg.hint_transformation)

    # Prepare evaluation tracking
    iteration_tracking: List[evaluation.IterationResult] = []
    compute_iteration_metrics = get_compute_metrics(
        iteration_tracking=iteration_tracking,
        average=cfg.experiment.average,
        run_name=run_name,
        id2label=id2label,
    )

    # Start kfold
    for iteration, dataset_paths in split_dataset_k_fold(
        name=run_name,
        dataset=transformed_dataset["dataset"],
        folds=cfg.experiment.folds,
        random_state=cfg.experiment.random_state,
    ):
        log_artifacts(dataset_paths)

        t_a_a_l = get_tokenize_and_align_labels(
            tokenizer=TOKENIZER, max_len=max_len, truncation=True
        )

        # Load training data and create a training file
        data_train = load_split_dataset(
            name=run_name, filename=DATA_TRAIN, iteration=iteration
        )
        train_data = create_tore_dataset(data=data_train, label2id=label2id)
        train_data = train_data.add_column(
            "hint_input_ids",
            generate_hint_data(
                train_data["tore_label_id"],
                id2label=id2label,
                hint_transformation=hints["transformation_function"],
                hint_label2id=hints["label2id"],
            ),
        )

        train_data = train_data.map(t_a_a_l, batched=True)

        train_data = train_data.rename_columns(
            {"string": "text", "tore_label_id": "labels"}
        )

        # Create Solution
        data_test = load_split_dataset(
            name=run_name, filename=DATA_TEST, iteration=iteration
        )

        test_data = create_tore_dataset(data=data_test, label2id=label2id)
        test_data = test_data.add_column(
            "hint_input_ids",
            generate_hint_data(
                test_data["tore_label_id"],
                id2label=id2label,
                hint_transformation=hints["transformation_function"],
                hint_label2id=hints["label2id"],
            ),
        )

        test_data = test_data.map(t_a_a_l, batched=True)

        test_data = test_data.rename_columns(
            {"string": "text", "tore_label_id": "labels"}
        )

        # Get Model

        model = StagedBertForTokenClassification.from_pretrained(
            cfg.bert.model,
            num_hint_labels=len(hints["label2id"]),
            num_labels=len(transformed_dataset["labels"]),
            id2label=id2label,
            label2id=label2id,
        )
        # for param in model.bert.parameters():
        #    param.requires_grad = False

        model.to(device)

        compute_metrics = get_compute_metrics(
            iteration_tracking=[],
            average=cfg.experiment.average,
            run_name=run_name,
            id2label=id2label,
        )

        model_dir = model_path(name=run_name, iteration=iteration)

        output_dir = str(
            output_path(name=run_name, clean=True)
        )  # pin iteration to avoid relogging parameter

        logging_dir = str(output_path(name=run_name, clean=True))

        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=logging_dir,
            run_name=run_name,
            learning_rate=cfg.bert.learning_rate,
            per_device_train_batch_size=cfg.bert.train_batch_size,
            per_device_eval_batch_size=cfg.bert.validation_batch_size,
            num_train_epochs=cfg.bert.number_epochs,
            weight_decay=cfg.bert.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            optim="adamw_torch",
            push_to_hub=False,
            use_mps_device=True,
        )

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=TOKENIZER,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
        )
        trainer.train()
        trainer.save_model(output_dir=str(model_dir))

        test_results = trainer.predict(test_data)

        compute_iteration_metrics((test_results[0], test_results[1]))
        log_iteration_result(result=iteration_tracking[-1])
        log_artifacts(model_dir)
    # Evaluate Run

    experiment_result = evaluation.evaluate_experiment(
        run_name=run_name, iteration_results=iteration_tracking
    )
    log_experiment_result(result=experiment_result)

    end_tracing()


if __name__ == "__main__":
    main()
