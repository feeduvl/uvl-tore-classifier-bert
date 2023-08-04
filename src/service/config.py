import os
import pickle
from pathlib import Path

from flask import g
from transformers import BertTokenizerFast

from classifiers.bilstm import get_glove_model
from service.types import Label2Id2Label
from service.types import Models


def configure() -> None:
    if "glove_model" not in g:
        g.glove_model = get_glove_model()

    if "models" not in g:
        g.models = get_models()

    if "label2id2label" not in g:
        g.label2id2label = get_label2id2label()

    if "tokenizer" not in g:
        g.tokenizer = get_tokenizer()

    if "max_len" not in g:
        g.max_len = get_max_len()


def get_tokenizer() -> BertTokenizerFast:
    return BertTokenizerFast.from_pretrained(
        os.environ.get("UVL_BERT_BASE_MODEL", "bert-base-uncased")
    )


def get_max_len() -> int:
    return int(os.environ.get("UVL_MAX_LEN", "106"))


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
