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
                .joinpath(Path("./sner.ser.gz")),
            )
        ),
        bilstm=Path(
            os.environ.get(
                "UVL_BILSTM_MODEL",
                Path(__file__).parent.resolve().joinpath(Path("./bilstm")),
            )
        ),
        bert_1=Path(
            os.environ.get(
                "UVL_BERT_1_MODEL",
                Path(__file__).parent.resolve().joinpath(Path("./bert_1")),
            )
        ),
        bert_2=Path(
            os.environ.get(
                "UVL_BERT_2_MODEL",
                Path(__file__).parent.resolve().joinpath(Path("bert_2")),
            )
        ),
        bert=Path(
            os.environ.get(
                "UVL_BERT_MODEL",
                Path(__file__).parent.resolve().joinpath(Path("bert")),
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
                    .joinpath(Path("label2id.pickle")),
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
                    .joinpath(Path("id2label.pickle")),
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
                    .joinpath(Path("hint_label2id.pickle")),
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
                    .joinpath(Path("hint_id2label.pickle")),
                ),
                "rb",
            )
        ),
    )
