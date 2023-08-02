from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import TypedDict

from tooling.model import Label_None_Pad

Documents = List[Dict[Literal["text"], str]]


class Code(TypedDict):
    tokens: List[int]
    name: str
    tore: Label_None_Pad
    index: int
    relationship_memberships: List[None]


class Token(TypedDict):
    index: int
    name: str
    lemma: str
    pos: str
    num_name_codes: int
    num_tore_codes: int


class Annotation(TypedDict):
    uploaded_at: datetime
    last_updated: datetime

    name: str
    dataset: str

    tores: List[str]
    show_recommendationtore: bool
    docs: List[Documents]
    tokens: List[Token]
    codes: List[Code]
    tore_relationships: List[Any]


@dataclass
class Models:
    sner: Path
    bilstm: Path
    bert_1: Path
    bert_2_bert: Path
    bert_2_sner: Path
    bert_2_bilstm: Path
    bert: Path


@dataclass
class Label2Id2Label:
    label2id: Dict[Label_None_Pad, int]
    id2label: Dict[int, Label_None_Pad]
    hint_label2id: Dict[Label_None_Pad, int]
    hint_id2label: Dict[int, Label_None_Pad]


Classifier_Options = Literal["bilstm_bert", "sner_bert", "bert_bert", "bert"]
