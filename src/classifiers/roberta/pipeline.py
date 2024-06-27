from typing import List, Any, Dict

import mlflow
import torch
from transformers import RobertaForTokenClassification
from transformers import RobertaTokenizerFast
from transformers import TrainingArguments
from datasets import Dataset
from classifiers.roberta.classifier import create_roberta_dataset
from classifiers.roberta.classifier import get_compute_metrics
from classifiers.roberta.classifier import get_max_len
from classifiers.roberta.classifier import setup_device
from classifiers.roberta.classifier import WeightedTrainer
from classifiers.roberta.files import model_path
from classifiers.roberta.files import output_path
from tooling import evaluation
from tooling.config import RoBERTaConfig
from tooling.loading import import_dataset
from tooling.logging import logging_setup
from tooling.model import get_id2label
from tooling.model import get_label2id, Label_None_Pad
from tooling.observability import log_artifacts
from tooling.sampling import DATA_TEST
from tooling.sampling import DATA_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset_k_fold
from tooling.transformation import get_class_weights
from tooling.transformation import transform_dataset, TransformedDataset
from tooling.types import IterationResult


logging = logging_setup(__name__)


def roberta_pipeline(cfg: RoBERTaConfig, run_name: str) -> None:
    device = setup_device()

    # Import Dataset
    import_dataset(cfg, run_name)

    # Setup Tokenizer
    TOKENIZER = RobertaTokenizerFast.from_pretrained(cfg.roberta.model, add_prefix_space=True, ignore_mismatched_sizes=True)

    # Transform Dataset
    transformed_dataset = transform_dataset(
        cfg, run_name, fill_with_zeros=True
    )

    # Get model constants
    class_weights = torch.from_numpy(
        get_class_weights(
            exp_cfg=cfg.roberta, data=transformed_dataset["dataset"]
        )
    ).to(torch.float32)

    id2label = get_id2label(transformed_dataset["labels"])
    label2id = get_label2id(transformed_dataset["labels"])
    mlflow.log_param("id2label", id2label)
    mlflow.log_param("label2id", label2id)

    max_len = get_max_len(
        roberta_cfg=cfg.roberta,
        data=transformed_dataset["dataset"],
        label2id=label2id,
        tokenizer=TOKENIZER,
    )

    # Prepare evaluation tracking
    iteration_tracking: List[IterationResult] = []
    compute_iteration_metrics = get_compute_metrics(
        iteration_tracking=iteration_tracking,
        average=cfg.experiment.average,
        run_name=run_name,
        labels=transformed_dataset["labels"],
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
        train_data = create_roberta_dataset(
            input_data=load_split_dataset(
                name=run_name, filename=DATA_TRAIN, iteration=iteration
            ),
            label2id=label2id,
            tokenizer=TOKENIZER,
            max_len=max_len,
        )

        # Load testing data
        test_data = create_roberta_dataset(
            input_data=load_split_dataset(
                name=run_name, filename=DATA_TEST, iteration=iteration
            ),
            label2id=label2id,
            tokenizer=TOKENIZER,
            max_len=max_len,
        )

        # Get Model
        trainer = train_roberta(
            cfg,
            run_name,
            device,
            TOKENIZER,
            transformed_dataset,
            class_weights,
            id2label,
            label2id,
            iteration,
            train_data,
            test_data,
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


def train_roberta(
    cfg: RoBERTaConfig,
    run_name: str,
    device: str,
    TOKENIZER: Any,
    transformed_dataset: TransformedDataset,
    class_weights: torch.Tensor,
    id2label: Dict[int, Label_None_Pad],
    label2id: Dict[Label_None_Pad, int],
    iteration: int,
    train_data: Dataset,
    test_data: Dataset,
) -> WeightedTrainer:
    model = RobertaForTokenClassification.from_pretrained(
        cfg.roberta.model,
        num_labels=len(transformed_dataset["labels"]),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    model.to(device=device)

    # Train

    training_args = TrainingArguments(
        output_dir=str(
            output_path(name=run_name, clean=True)
        ),  # pin iteration to avoid relogging parameter,
        logging_dir=str(
            output_path(name=run_name, clean=True)
        ),  # pin iteration to avoid relogging parameter,
        run_name=run_name,
        per_device_train_batch_size=cfg.roberta.train_batch_size,
        per_device_eval_batch_size=cfg.roberta.validation_batch_size,
        num_train_epochs=cfg.roberta.number_epochs,
        weight_decay=cfg.roberta.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        optim="adamw_torch",
        push_to_hub=False,
        use_mps_device=(device == "mps"),
    )

    # mypy: allow-untyped-call
    trainer = WeightedTrainer(  # type: ignore
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=TOKENIZER,
        compute_metrics=get_compute_metrics(
            iteration_tracking=[],
            average=cfg.experiment.average,
            labels=transformed_dataset["labels"],
            run_name=run_name,
            id2label=id2label,
            create_confusion_matrix=False,
        ),
        class_weights=class_weights,
        device=device,
        learning_rate_roberta=cfg.roberta.learning_rate_roberta,
        learning_rate_classifier=cfg.roberta.learning_rate_classifier,
    )
    trainer.train()
    trainer.save_model(
        output_dir=str(model_path(name=run_name, iteration=iteration))
    )

    return trainer
