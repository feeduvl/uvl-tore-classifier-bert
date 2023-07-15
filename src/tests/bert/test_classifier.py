import uuid
from collections.abc import Sequence
from typing import cast
from typing import get_args
from typing import Literal

import numpy as np
import pandas as pd
import pytest
from strictly_typed_pandas import DataSet
from transformers import BertTokenizerFast

from classifiers.bert.classifier import BertData
from classifiers.bert.classifier import tokenize_and_align_labels
from tooling.model import DataDF
from tooling.model import get_labels
from tooling.model import Label_None_Pad
from tooling.model import Token
from tooling.model import tokenlist_to_datadf
from tooling.model import TORE_LABELS_NONE_PAD


def test_tokenize_and_align_labels():
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

    result = tokenize_and_align_labels(data=data, tokenizer=tokenizer)

    assert len(result.encodings[0].tokens) == 22
