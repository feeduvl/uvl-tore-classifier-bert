import dataclasses
import itertools
import typing
import uuid
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from typing import cast
from typing import get_args
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import TypeAlias
from typing import Union

import pandas as pd
from pydantic import BaseModel
from pydantic.dataclasses import (
    dataclass,
)
from strictly_typed_pandas import DataSet

Pos = Literal["v", "n", "a", "r", ""]


ImportDomainLevelLabel = Literal[
    "Task",
    "Goals",
    "Domain Data",
    "Activity",
    "Stakeholder",
]
DomainLevelLabel = Literal[
    "Task",
    "Goals",
    "Domain_Data",
    "Activity",
    "Stakeholder",
]

DOMAIN_LEVEL_LABELS: Tuple[DomainLevelLabel, ...] = cast(
    tuple[DomainLevelLabel], typing.get_args(DomainLevelLabel)
)


ImportInteractionLevelLabel = Literal[
    "System Function",
    "Interaction",
    "Interaction Data",
    "Workspace",
]
InteractionLevelLabel = Literal[
    "System_Function",
    "Interaction",
    "Interaction_Data",
    "Workspace",
]
INTERACTION_LEVEL_LABELS: Tuple[InteractionLevelLabel, ...] = cast(
    tuple[InteractionLevelLabel], typing.get_args(InteractionLevelLabel)
)

ImportSystemLevelLabel = Literal[
    "Software",
    "Internal Action",
    "Internal Data",
]
SystemLevelLabel = Literal[
    "Software",
    "Internal_Action",
    "Internal_Data",
]
SYSTEM_LEVEL_LABELS: Tuple[SystemLevelLabel, ...] = cast(
    tuple[SystemLevelLabel], typing.get_args(SystemLevelLabel)
)

AdditionalLabel = Literal["Relationship"]
ADDITIONAL_LABEL: Tuple[AdditionalLabel, ...] = cast(
    tuple[AdditionalLabel], typing.get_args(AdditionalLabel)
)


ImportToreLabel = Literal[
    ImportSystemLevelLabel, ImportInteractionLevelLabel, ImportDomainLevelLabel
]
ToreLabel = Literal[SystemLevelLabel, InteractionLevelLabel, DomainLevelLabel]
TORE_LABELS = (
    DOMAIN_LEVEL_LABELS + INTERACTION_LEVEL_LABELS + SYSTEM_LEVEL_LABELS
)


DomainLevel = Literal["Domain_Level"]
DOMAIN_LEVEL: DomainLevel = typing.get_args(DomainLevel)[0]

InteractionLevel = Literal["Interaction_Level"]
INTERACTION_LEVEL: InteractionLevel = typing.get_args(InteractionLevel)[0]

SystemLevel = Literal["System_Level"]
SYSTEM_LEVEL: SystemLevel = typing.get_args(SystemLevel)[0]

ToreLevel = Literal[DomainLevel, InteractionLevel, SystemLevel]
TORE_LEVEL: Tuple[ToreLevel, ...] = (
    DOMAIN_LEVEL,
    INTERACTION_LEVEL,
    SYSTEM_LEVEL,
)

Domain: TypeAlias = Tuple[DomainLevel, Tuple[DomainLevelLabel, ...]]
Interaction: TypeAlias = Tuple[
    InteractionLevel, Tuple[InteractionLevelLabel, ...]
]
System: TypeAlias = Tuple[SystemLevel, Tuple[SystemLevelLabel, ...]]

DOMAIN: Domain = (DOMAIN_LEVEL, DOMAIN_LEVEL_LABELS)
INTERACTION: Interaction = (INTERACTION_LEVEL, INTERACTION_LEVEL_LABELS)
SYSTEM: System = (SYSTEM_LEVEL, SYSTEM_LEVEL_LABELS)

Tore: TypeAlias = tuple[Domain, Interaction, System]
TORE: Tore = (
    DOMAIN,
    INTERACTION,
    SYSTEM,
)

Label = Literal[ToreLabel, ToreLevel, Literal["0"]]
Label_Pad = Label | Literal["_"]

PAD: Literal["_"] = "_"

TORE_LABELS_0: Tuple[Label, ...] = TORE_LABELS + ("0",)
TORE_LABELS_0_PAD: Tuple[Label_Pad, ...] = TORE_LABELS_0 + (PAD,)

LABELS_0: Tuple[Label, ...] = TORE_LABELS + TORE_LEVEL + ("0",)
LABELS_0_PAD: Tuple[Label_Pad, ...] = TORE_LABELS_0 + TORE_LEVEL + (PAD,)


class ImportDoc(BaseModel):
    name: str
    begin_index: int
    end_index: int


class ImportToken(BaseModel):
    index: Optional[int]
    name: str
    lemma: str
    pos: Pos
    num_name_codes: int
    num_tore_codes: int


class ImportCode(BaseModel):
    index: Optional[int]
    tokens: List[int]
    name: str
    tore: Union[ImportToreLabel, Literal[""], AdditionalLabel]


class ImportDataSet(BaseModel):
    uploaded_at: datetime
    last_updated: datetime
    name: str
    dataset: str
    docs: List[ImportDoc]
    tokens: List[ImportToken]
    codes: List[ImportCode]


@dataclass(kw_only=True)
class Token:
    sentence_id: Optional[uuid.UUID]
    sentence_idx: Optional[int]

    string: str
    lemma: str
    pos: Pos

    source: str

    tore_label: Optional[Label]


class DataDF:
    id: int
    sentence_id: uuid.UUID
    sentence_idx: int
    string: str
    tore_label: Optional[Label]


class SentencesDF:
    id: int
    sentences_id: uuid.UUID


class ResultDF:
    id: int
    string: str
    tore_label: Optional[Label]


def create_datadf(data: pd.DataFrame) -> DataSet[DataDF]:
    datadf = data[["sentence_id", "sentence_idx", "string", "tore_label"]]
    return cast(DataSet[DataDF], data)


def data_to_sentences(data: DataSet[DataDF]) -> List[str]:
    sentences: List[str] = []

    for sentence_idx, grouped_data in data.groupby("sentence_id"):
        sentence = " ".join([string for string in grouped_data["string"]])
        sentences.append(sentence)

    return sentences


def data_to_list_of_token_lists(
    data: DataSet[DataDF],
) -> List[List[str]]:
    sentences: List[List[str]] = []

    for sentence_idx, grouped_data in data.groupby("sentence_id"):
        sentence = [string for string in grouped_data["string"]]
        sentences.append(sentence)

    return sentences


def data_to_list_of_label_lists(
    data: DataSet[DataDF],
) -> List[List[Label]]:
    sentences: List[List[Label]] = []

    for sentence_idx, grouped_data in data.groupby("sentence_id"):
        sentence = [
            cast(Label, string) for string in grouped_data["tore_label"]
        ]
        sentences.append(sentence)

    return sentences


def get_labels(
    dataset: DataSet[ResultDF] | DataSet[DataDF],
) -> List[Label]:
    return cast(List[Label], dataset["tore_label"].unique().tolist())


def tokenlist_to_datadf(tokens: List[Token]) -> DataSet[DataDF]:
    dataframe = pd.DataFrame(tokens)
    return cast(DataSet[DataDF], dataframe)
