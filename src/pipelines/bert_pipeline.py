import sys
from typing import List

import hydra
import torch
from hydra.core.config_store import ConfigStore
from transformers import BertForTokenClassification
from transformers import BertTokenizerFast
from transformers import TrainingArguments

from classifiers.bert.classifier import create_bert_dataset
from classifiers.bert.classifier import get_compute_metrics
from classifiers.bert.classifier import get_max_len
from classifiers.bert.classifier import setup_device
from classifiers.bert.classifier import WeightedTrainer
from classifiers.bert.files import model_path
from classifiers.bert.files import output_path
from tooling import evaluation
from tooling.config import BERTConfig
from tooling.loading import import_dataset
from tooling.logging import logging_setup
from tooling.model import get_id2label
from tooling.model import get_label2id
from tooling.observability import config_mlflow
from tooling.observability import end_tracing
from tooling.observability import log_artifacts
from tooling.observability import RerunException
from tooling.sampling import DATA_TEST
from tooling.sampling import DATA_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset_k_fold
from tooling.transformation import get_class_weights
from tooling.transformation import transform_dataset
from tooling.types import IterationResult

cs = ConfigStore.instance()
cs.store(name="base_config", node=BERTConfig)

logging = logging_setup(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config_bert")
def main(cfg: BERTConfig) -> None:
    run_name = config_mlflow(cfg)
    device = setup_device()

    # Import Dataset
    import_dataset(cfg, run_name)

    # Setup Tokenizer
    TOKENIZER = BertTokenizerFast.from_pretrained(cfg.bert.model)

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
        )

        # Load testing data
        test_data = create_bert_dataset(
            input_data=load_split_dataset(
                name=run_name, filename=DATA_TEST, iteration=iteration
            ),
            label2id=label2id,
            tokenizer=TOKENIZER,
            max_len=max_len,
        )

        # Get Model
        model = BertForTokenClassification.from_pretrained(
            cfg.bert.model,
            num_labels=len(transformed_dataset["labels"]),
            id2label=id2label,
            label2id=label2id,
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
            ),  # pin iteration to avoid relogging parameter,
            run_name=run_name,
            per_device_train_batch_size=cfg.bert.train_batch_size,
            per_device_eval_batch_size=cfg.bert.validation_batch_size,
            num_train_epochs=cfg.bert.number_epochs,
            weight_decay=cfg.bert.weight_decay,
            evaluation_strategy="steps",
            eval_steps=1,
            save_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            optim="adamw_torch",
            push_to_hub=False,
            use_mps_device=True,
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
                run_name=run_name,
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
        if iteration == cfg.experiment.iterations:
            logging.info(
                f"Breaking early after {iteration=} of {cfg.experiment.folds} folds"
            )
            break

    # Evaluate Run

    evaluation.evaluate_experiment(
        run_name=run_name, iteration_results=iteration_tracking
    )

    end_tracing(status="FINISHED")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        logging.info("Keyobard interrupt recieved")
        status = "FAILED"
        end_tracing(status=status)

        sys.exit()

    except Exception as e:
        logging.error(e)
        status = "FAILED"
        end_tracing(status=status)

        raise e
