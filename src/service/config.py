import os
import pickle
from pathlib import Path


from transformers import BertTokenizerFast

from typing import Dict, Any
from classifiers.bilstm import get_glove_model
from service.service_types import Label2Id2Label
from service.service_types import Models


def configure(cache: Dict[str, Any]) -> Dict[str, Any]:
    if cache.get("glove_model", None) is None:
        cache["glove_model"] = get_glove_model()

    if cache.get("models", None) is None:
        cache["models"] = get_models()

    if cache.get("label2id2label", None) is None:
        cache["label2id2label"] = get_label2id2label()

    if cache.get("max_len", None) is None:
        cache["max_len"] = get_max_len()

    return cache


def get_max_len() -> int:
    return int(os.environ.get("UVL_MAX_LEN", "110"))


def get_models() -> Models:
    return Models(
        sner=Path(
            os.environ.get(
                "UVL_SNER_MODEL",
                Path(__file__)
                .parent.resolve()
                .joinpath(Path("./models/sner.ser.gz")),
            )
        ),
        bilstm=Path(
            os.environ.get(
                "UVL_BILSTM_MODEL",
                Path(__file__)
                .parent.resolve()
                .joinpath(Path("./models/bilstm")),
            )
        ),
        bert_1=Path(
            os.environ.get(
                "UVL_BERT_1_MODEL",
                Path(__file__)
                .parent.resolve()
                .joinpath(Path("./models/bert_1")),
            )
        ),
        bert_2_bert=Path(
            os.environ.get(
                "UVL_BERT_2_BERT_MODEL",
                Path(__file__)
                .parent.resolve()
                .joinpath(Path("./models/bert_2_bert")),
            )
        ),
        bert_2_sner=Path(
            os.environ.get(
                "UVL_BERT_2_SNER_MODEL",
                Path(__file__)
                .parent.resolve()
                .joinpath(Path("./models/bert_2_sner")),
            )
        ),
        bert_2_bilstm=Path(
            os.environ.get(
                "UVL_BERT_2_BILSTM_MODEL",
                Path(__file__)
                .parent.resolve()
                .joinpath(Path("./models/bert_2_bilstm")),
            )
        ),
        bert=Path(
            os.environ.get(
                "UVL_BERT_MODEL",
                Path(__file__)
                .parent.resolve()
                .joinpath(Path("./models/bert")),
            )
        ),
    )


def get_label2id2label() -> Label2Id2Label:
    return Label2Id2Label(
        label2id=pickle.load(
            open(
                os.environ.get(
                    "UVL_LABEL2ID",
                    Path(__file__)
                    .parent.resolve()
                    .joinpath(Path("./models/label2id.pickle")),
                ),
                "rb",
            )
        ),
        id2label=pickle.load(
            open(
                os.environ.get(
                    "UVL_ID2LABEL",
                    Path(__file__)
                    .parent.resolve()
                    .joinpath(Path("./models/id2label.pickle")),
                ),
                "rb",
            )
        ),
        hint_label2id=pickle.load(
            open(
                os.environ.get(
                    "UVL_HINT_LABEL2ID",
                    Path(__file__)
                    .parent.resolve()
                    .joinpath(Path("./models/hint_label2id.pickle")),
                ),
                "rb",
            )
        ),
        hint_id2label=pickle.load(
            open(
                os.environ.get(
                    "UVL_HINT_ID2LABEL",
                    Path(__file__)
                    .parent.resolve()
                    .joinpath(Path("./models/hint_id2label.pickle")),
                ),
                "rb",
            )
        ),
    )


if __name__ == "__main__":
    configure()
