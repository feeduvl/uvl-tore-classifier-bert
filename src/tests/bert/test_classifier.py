import pytest
from transformers import BertTokenizerFast

from classifiers.bert.classifier import BertData
from classifiers.bert.classifier import BertDatas
from classifiers.bert.classifier import tokenize_and_align_labels


def test_tokenize_and_align_labels() -> None:
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokens = [
        "This",
        "is",
        "a",
        "contrieved",
        "example",
        "using",
        "elaborate",
        "words",
        "to",
        "force",
        "subtoken",
        "splitting",
        ",",
        "nice",
        "!",
    ]

    labels = [
        0,
        1,
        2,
        3,
        0,
        0,
        1,
        2,
        3,
        0,
        0,
        1,
        2,
        3,
        0,
    ]
    data: BertData = {"id": 1, "string": tokens, "tore_label_id": labels}

    datas: BertDatas = {
        "id": [data["id"]],
        "string": [data["string"]],
        "tore_label_id": [data["tore_label_id"]],
    }

    result = tokenize_and_align_labels(
        data=datas, tokenizer=tokenizer, max_len=22, align_labels=False
    )

    assert len(result.encodings[0].tokens) == 22
