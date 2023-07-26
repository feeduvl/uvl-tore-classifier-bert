import typing
import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import cast
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import overload
from typing import Tuple
from typing import TypeAlias
from typing import Union

import pandas as pd
from pydantic import BaseModel
from pydantic.dataclasses import (
    dataclass,
)
from strictly_typed_pandas.dataset import DataSet

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

NoneLabel = Literal["0"]
Pad = Literal["_"]

ZERO: NoneLabel = "0"
PAD: Pad = "_"

Label = Literal[ToreLabel, ToreLevel]
Label_None = Literal[ToreLabel, ToreLevel, NoneLabel]
Label_None_Pad = Label_None | Literal["_"]

TORE_LABELS = (
    DOMAIN_LEVEL_LABELS + INTERACTION_LEVEL_LABELS + SYSTEM_LEVEL_LABELS
)
TORE_LABELS_NONE: Tuple[Label_None, ...] = (ZERO,) + TORE_LABELS
TORE_LABELS_NONE_PAD: Tuple[Label_None_Pad, ...] = TORE_LABELS_NONE + (PAD,)

LABELS: Tuple[Label, ...] = TORE_LABELS + TORE_LEVEL
LABELS_NONE: Tuple[Label_None, ...] = (ZERO,) + TORE_LABELS + TORE_LEVEL
LABELS_NONE_PAD: Tuple[Label_None_Pad, ...] = (
    (ZERO,) + TORE_LABELS + TORE_LEVEL + (PAD,)
)


def label_to_id(label: Label_None_Pad) -> int:
    try:
        return LABELS_NONE.index(label)
    except ValueError as e:
        if label == PAD:
            return -1
        else:
            raise e


def id_to_label(label_id: int) -> Label_None_Pad:
    if label_id == -1:
        return PAD

    return LABELS_NONE[label_id]


def get_label2id(
    labels: Sequence[Label_None_Pad],
) -> Dict[Label_None_Pad, int]:
    sorted_labels = list(set(labels))
    sorted_labels.sort(key=lambda x: LABELS_NONE_PAD.index(x))
    return {label: idx for idx, label in enumerate(sorted_labels)}


def get_id2label(
    labels: Sequence[Label_None_Pad],
) -> Dict[int, Label_None_Pad]:
    sorted_labels = list(set(labels))
    sorted_labels.sort(key=lambda x: LABELS_NONE_PAD.index(x))
    return {idx: label for idx, label in enumerate(sorted_labels)}


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

    tore_label: Optional[Label | Pad | NoneLabel]


@dataclass
class DataDF:
    id: int
    sentence_id: uuid.UUID
    sentence_idx: int
    string: str
    tore_label: Optional[Label]


TokenizedDataDF = typing.NewType("TokenizedDataDF", DataDF)


class SentencesDF:
    id: int
    sentences_id: uuid.UUID


@dataclass
class ResultDF:
    id: Optional[int]
    string: str
    tore_label: Optional[Label]


@dataclass
class ToreLabelDF:
    id: Optional[int]
    tore_label: Optional[Label]


def create_datadf(data: pd.DataFrame) -> DataSet[DataDF]:
    datadf = data[["sentence_id", "sentence_idx", "string", "tore_label"]]
    return cast(DataSet[DataDF], datadf)


def create_resultdf(data: pd.DataFrame) -> DataSet[ResultDF]:
    datadf = data[["string", "tore_label"]]
    return cast(DataSet[ResultDF], datadf)


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


def get_sentence_lengths(data: DataSet[DataDF]) -> List[int]:
    sentences = data_to_list_of_token_lists(data=data)
    lengths = [len(sentence) for sentence in sentences]
    return lengths


@overload
def data_to_list_of_label_lists(
    data: DataSet[DataDF],
    label2id: None,
    column: str = "tore_label",
) -> List[List[Label_None_Pad]]:
    ...


@overload
def data_to_list_of_label_lists(
    data: DataSet[DataDF],
    label2id: Dict[Label_None_Pad, int],
    column: str = "tore_label",
) -> List[List[int]]:
    ...


def data_to_list_of_label_lists(
    data: DataSet[DataDF],
    label2id: Dict[Label_None_Pad, int] | None,
    column: str = "tore_label",
) -> Union[List[List[Label_None_Pad]], List[List[int]]]:
    if label2id:
        id_sentences = []
        for sentence_idx, grouped_data in data.groupby("sentence_id"):
            sentence_id = [label2id[string] for string in grouped_data[column]]
            id_sentences.append(sentence_id)
        return id_sentences
    else:
        label_sentences = []
        for sentence_idx, grouped_data in data.groupby("sentence_id"):
            sentence_label = [
                cast(Label_None_Pad, string) for string in grouped_data[column]
            ]
            label_sentences.append(sentence_label)
        return label_sentences


def get_labels(
    dataset: DataSet[ResultDF] | DataSet[DataDF] | DataSet[ToreLabelDF],
) -> List[Label]:
    return cast(List[Label], dataset["tore_label"].unique().tolist())


def tokenlist_to_datadf(tokens: List[Token]) -> DataSet[DataDF]:
    dataframe = pd.DataFrame(tokens)
    return cast(DataSet[DataDF], dataframe)
