# %%
import hydra
import mlflow
import pandas as pd
from classifiers.sner import classify_sentences
from classifiers.sner import create_config_file
from classifiers.sner import create_train_file
from classifiers.sner import load_classification_result
from classifiers.sner import sentences_to_token_df
from classifiers.sner import train_sner
from data import return_dataset
from omegaconf import DictConfig
from omegaconf import OmegaConf
from tooling import evaluation
from tooling.loading import import_dataset
from tooling.loading import load_dataset
from tooling.model import TORE_LABELS
from tooling.sampling import LABELS_TEST
from tooling.sampling import LABELS_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset
from tooling.sampling import TEXT_TEST


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # %%
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.autolog()

    print(OmegaConf.to_yaml(cfg))
    df = pd.json_normalize(OmegaConf.to_container(cfg), sep="_")
    mlflow.log_params(params=df.to_dict(orient="records")[0])

    # %%

    dataset = return_dataset(cfg.dataset.name)
    import_dataset(name=cfg.experiment.name, ds_spec=dataset)

    # %%

    d = load_dataset(name=cfg.experiment.name)

    split_dataset(
        name=cfg.experiment.name,
        text=d["text"],
        labels=d["self"],
        test_size=cfg.experiment.test_size,
        stratify=d["source"],
        random_state=cfg.experiment.random_state,
    )

    # %%

    labels_train = load_split_dataset(
        name=cfg.experiment.name, filename=LABELS_TRAIN
    )

    create_config_file(name=cfg.experiment.name)
    create_train_file(name=cfg.experiment.name, sentences=labels_train)

    train_sner(name=cfg.experiment.name)

    # %%

    text_test = load_split_dataset(
        name=cfg.experiment.name, filename=TEXT_TEST
    )
    classify_sentences(name=cfg.experiment.name, sentences=text_test)

    # %%

    results = load_classification_result(name=cfg.experiment.name)

    labels_test = load_split_dataset(
        name=cfg.experiment.name, filename=LABELS_TEST
    )
    solution = sentences_to_token_df(labels_test)

    # %%

    p = evaluation.score_precision(
        solution=solution["label"],
        results=results["label"],
        labels=TORE_LABELS,
    )
    mlflow.log_metric("precision", p)

    # %%

    r = evaluation.score_recall(
        solution=solution["label"],
        results=results["label"],
        labels=TORE_LABELS,
    )
    mlflow.log_metric("recall", r)

    # %%

    conf = evaluation.confusion_matrix(
        name=cfg.experiment.name,
        solution=solution["label"],
        results=results["label"],
    )
    mlflow.log_artifact(conf)


if __name__ == "__main__":
    main()
