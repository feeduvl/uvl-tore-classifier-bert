from typing import List

import mlflow
import torch
from transformers import BertTokenizerFast
from transformers import TrainingArguments

from classifiers.bert.classifier import create_bert_dataset
from classifiers.bert.classifier import get_compute_metrics
from classifiers.bert.classifier import get_max_len
from classifiers.bert.classifier import setup_device
from classifiers.bert.classifier import WeightedTrainer
from classifiers.bert.files import model_path
from classifiers.bert.files import output_path
from classifiers.staged_bert.classifier import get_hint_modifier
from classifiers.staged_bert.classifier import get_hint_transformation
from classifiers.staged_bert.model import (
    StagedBertForTokenClassification,
    StagedBertModelConfig,
)
from tooling import evaluation
from tooling.config import DualModelStagedBERTConfig
from tooling.config import StagedBERTConfig
from tooling.experiment import get_model
from tooling.experiment import run_model
from tooling.loading import import_dataset
from tooling.logging import logging_setup
from tooling.model import get_id2label
from tooling.model import get_label2id
from tooling.observability import end_tracing
from tooling.observability import log_artifacts
from tooling.sampling import DATA_TEST
from tooling.sampling import DATA_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset_k_fold
from tooling.transformation import get_class_weights
from tooling.transformation import transform_dataset
from tooling.types import IterationResult


logging = logging_setup(__name__)


def staged_bert_pipeline(cfg: StagedBERTConfig, run_name: str) -> None:
    device = setup_device()

    mlflow.log_param("bert.num_layers", len(cfg.bert.layers))
    mlflow.log_param("bert.layers", ",".join(str(x) for x in cfg.bert.layers))

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
    class_weights = torch.from_numpy(
        get_class_weights(
            exp_cfg=cfg.bert, data=transformed_dataset["dataset"]
        )
    ).to(torch.float32)
    id2label = get_id2label(transformed_dataset["labels"])
    label2id = get_label2id(transformed_dataset["labels"])

    max_len = get_max_len(
        bert_cfg=cfg.bert,
        data=transformed_dataset["dataset"],
        label2id=label2id,
        tokenizer=TOKENIZER,
    )

    hints = get_hint_transformation(cfg.hint_transformation)
    mlflow.log_param("hint_label2id", hints["label2id"])
    hint_modifier = get_hint_modifier(id2label=id2label, hints=hints)

    # Prepare evaluation tracking
    iteration_tracking: List[IterationResult] = []
    compute_iteration_metrics = get_compute_metrics(
        iteration_tracking=iteration_tracking,
        average=cfg.experiment.average,
        labels=transformed_dataset["labels"],
        run_name=run_name,
        id2label=id2label,
        create_confusion_matrix=True,
    )

    # Start kfold
    for iteration, dataset_paths in split_dataset_k_fold(
        name=run_name,
        dataset=transformed_dataset["dataset"],
        cfg_experiment=cfg.experiment,
    ):
        logging.info(f"Starting {iteration=}")

        # Load training data
        train_data = create_bert_dataset(
            input_data=load_split_dataset(
                name=run_name, filename=DATA_TRAIN, iteration=iteration
            ),
            label2id=label2id,
            tokenizer=TOKENIZER,
            max_len=max_len,
            modifiers=[hint_modifier],
        )

        # Load testing data
        test_data = create_bert_dataset(
            input_data=load_split_dataset(
                name=run_name, filename=DATA_TEST, iteration=iteration
            ),
            label2id=label2id,
            tokenizer=TOKENIZER,
            max_len=max_len,
            modifiers=[hint_modifier],
        )

        model_config = StagedBertModelConfig(
            num_hint_labels=len(hints["label2id"]),
            layers=cfg.bert.layers,
            num_labels=len(transformed_dataset["labels"]),
            id2label=id2label,
            label2id=label2id,
        )

        # Get Model
        model = StagedBertForTokenClassification.from_pretrained(
            cfg.bert.model, config=model_config
        )
        # for param in model.bert.parameters():
        #    param.requires_grad = False

        model.to(device=device)

        # Train

        training_args = TrainingArguments(
            output_dir=str(
                output_path(name=run_name, clean=True)
            ),  # pin iteration to avoid relogging parameter,
            logging_dir=str(
                output_path(name=run_name, clean=True)
            ),  # pin iteration to avoid relogging parameter
            run_name=run_name,
            per_device_train_batch_size=cfg.bert.train_batch_size,
            per_device_eval_batch_size=cfg.bert.validation_batch_size,
            num_train_epochs=cfg.bert.number_epochs,
            weight_decay=cfg.bert.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            optim="adamw_torch",
            push_to_hub=False,
        )

        trainer = WeightedTrainer(  # type: ignore
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=TOKENIZER,
            compute_metrics=get_compute_metrics(
                iteration_tracking=[],
                average=cfg.experiment.average,
                run_name=run_name,
                labels=transformed_dataset["labels"],
                id2label=id2label,
                create_confusion_matrix=False,
            ),
            class_weights=class_weights,
            device=device,
            learning_rate_bert=cfg.bert.learning_rate_bert,
            learning_rate_classifier=cfg.bert.learning_rate_classifier,
        )
        trainer.train()
        trainer.save_model(
            output_dir=str(model_path(name=run_name, iteration=iteration))
        )

        test_results = trainer.predict(test_data)

        compute_iteration_metrics((test_results[0], test_results[1]))

        logging.info(f"Finished {iteration=}")
        logging.info("Logging model artifact (might take a while)")
        log_artifacts(model_path(name=run_name, iteration=iteration))

        # early break if configured
        if iteration + 1 == cfg.experiment.iterations:
            logging.info(
                f"Breaking early after {iteration=} of {cfg.experiment.folds} folds"
            )
            break
    # Evaluate Run

    evaluation.evaluate_experiment(
        run_name=run_name, iteration_results=iteration_tracking
    )

    end_tracing()


def dual_stage_bert_pipeline(
    cfg: DualModelStagedBERTConfig, run_name: str
) -> None:
    device = setup_device()

    # Get First stage model
    hint_label2id, hint_id2label, glove_model, padded_labels = get_model(
        cfg, run_name=run_name
    )

    mlflow.log_param("bert.num_layers", len(cfg.bert.layers))
    mlflow.log_param("bert.layers", ",".join(str(x) for x in cfg.bert.layers))

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
    class_weights = torch.from_numpy(
        get_class_weights(
            exp_cfg=cfg.bert, data=transformed_dataset["dataset"]
        )
    ).to(torch.float32)
    id2label = get_id2label(transformed_dataset["labels"])
    label2id = get_label2id(transformed_dataset["labels"])

    max_len = get_max_len(
        bert_cfg=cfg.bert,
        data=transformed_dataset["dataset"],
        label2id=label2id,
        tokenizer=TOKENIZER,
    )

    # Prepare evaluation tracking
    iteration_tracking: List[IterationResult] = []
    compute_iteration_metrics = get_compute_metrics(
        iteration_tracking=iteration_tracking,
        average=cfg.experiment.average,
        labels=transformed_dataset["labels"],
        run_name=run_name,
        id2label=id2label,
        create_confusion_matrix=True,
    )

    # Start kfold
    for iteration, dataset_paths in split_dataset_k_fold(
        name=run_name,
        dataset=transformed_dataset["dataset"],
        cfg_experiment=cfg.experiment,
    ):
        logging.info(f"Starting {iteration=}")

        # Load training data
        input_training_data = load_split_dataset(
            name=run_name, filename=DATA_TRAIN, iteration=iteration
        )

        # Add predicted hint labels to train_dataset
        train_data = run_model(
            cfg=cfg,
            run_name=run_name,
            data=input_training_data,
            label2id=label2id,
            tokenizer=TOKENIZER,
            max_len=max_len,
            hint_label2id=hint_label2id,
            hint_id2label=hint_id2label,
            glove_model=glove_model,
            padded_labels=padded_labels,
        )

        # Load testing data
        input_test_data = load_split_dataset(
            name=run_name, filename=DATA_TEST, iteration=iteration
        )

        # Add predicted hint labels to test_dataset
        test_data = run_model(
            cfg=cfg,
            run_name=run_name,
            data=input_test_data,
            label2id=label2id,
            tokenizer=TOKENIZER,
            max_len=max_len,
            hint_label2id=hint_label2id,
            hint_id2label=hint_id2label,
            glove_model=glove_model,
            padded_labels=padded_labels,
        )

        model_config = StagedBertModelConfig(
            pretrained_model_name_or_path=cfg.bert.model,
            num_hint_labels=len(hint_label2id.keys()),
            layers=list(cfg.bert.layers),
            num_labels=len(transformed_dataset["labels"]),
            id2label=id2label,
            label2id=label2id,
        )

        # Get Model
        model = StagedBertForTokenClassification.from_pretrained(
            pretrained_model_name_or_path=cfg.bert.model,
            config=model_config,
            ignore_mismatched_sizes=True,
        )
        # for param in model.bert.parameters():
        #    param.requires_grad = False

        model.to(device=device)

        # Train
        training_args = TrainingArguments(
            output_dir=str(
                output_path(name=run_name, clean=True)
            ),  # pin iteration to avoid relogging parameter,
            logging_dir=str(
                output_path(name=run_name, clean=True)
            ),  # pin iteration to avoid relogging parameter
            run_name=run_name,
            per_device_train_batch_size=cfg.bert.train_batch_size,
            per_device_eval_batch_size=cfg.bert.validation_batch_size,
            num_train_epochs=cfg.bert.number_epochs,
            weight_decay=cfg.bert.weight_decay,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            optim="adamw_torch",
            push_to_hub=False,
        )

        trainer = WeightedTrainer(  # type: ignore
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            tokenizer=TOKENIZER,
            compute_metrics=get_compute_metrics(
                iteration_tracking=[],
                average=cfg.experiment.average,
                run_name=run_name,
                labels=transformed_dataset["labels"],
                id2label=id2label,
                create_confusion_matrix=False,
            ),
            class_weights=class_weights,
            device=device,
            learning_rate_bert=cfg.bert.learning_rate_bert,
            learning_rate_classifier=cfg.bert.learning_rate_classifier,
        )
        trainer.train()
        trainer.save_model(
            output_dir=str(model_path(name=run_name, iteration=iteration))
        )

        test_results = trainer.predict(test_data)

        compute_iteration_metrics((test_results[0], test_results[1]))

        logging.info(f"Finished {iteration=}")
        logging.info("Logging model artifact (might take a while)")
        log_artifacts(model_path(name=run_name, iteration=iteration))

        # early break if configured
        if iteration + 1 == cfg.experiment.iterations:
            logging.info(
                f"Breaking early after {iteration=} of {cfg.experiment.folds} folds"
            )
            break
    # Evaluate Run

    evaluation.evaluate_experiment(
        run_name=run_name, iteration_results=iteration_tracking
    )
