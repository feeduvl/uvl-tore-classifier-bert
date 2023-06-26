import dataclasses
import itertools
import typing
from collections import Counter
from datetime import datetime
from typing import cast
from typing import get_args
from typing import List
from typing import Literal
from typing import Optional
from typing import TypeAlias
from typing import Union

import pandas as pd
from pydantic import BaseModel
from pydantic.dataclasses import (
    dataclass,
)


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

DOMAIN_LEVEL_LABELS: tuple[DomainLevelLabel] = cast(
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
INTERACTION_LEVEL_LABELS: tuple[InteractionLevelLabel] = cast(
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
SYSTEM_LEVEL_LABELS: tuple[SystemLevelLabel] = cast(
    tuple[SystemLevelLabel], typing.get_args(SystemLevelLabel)
)

AdditionalLabel = Literal["Relationship"]
ADDITIONAL_LABEL: tuple[AdditionalLabel] = cast(
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
TORE_LEVEL = [DOMAIN_LEVEL, INTERACTION_LEVEL, SYSTEM_LEVEL]

Domain: TypeAlias = tuple[DomainLevel, tuple[DomainLevelLabel]]
Interaction: TypeAlias = tuple[InteractionLevel, tuple[InteractionLevelLabel]]
System: TypeAlias = tuple[SystemLevel, tuple[SystemLevelLabel]]

DOMAIN: Domain = (DOMAIN_LEVEL, DOMAIN_LEVEL_LABELS)
INTERACTION: Interaction = (INTERACTION_LEVEL, INTERACTION_LEVEL_LABELS)
SYSTEM: System = (SYSTEM_LEVEL, SYSTEM_LEVEL_LABELS)

Tore: TypeAlias = tuple[Domain, Interaction, System]
TORE: Tore = (
    DOMAIN,
    INTERACTION,
    SYSTEM,
)

# ToreLevelLabels:  List[tuple[ToreLevel, ToreLabel]]
# TORE_LEVEL_LABELS: ToreLevelLabels = cast(ToreLevelLabels,[
#    (DomainLevel, DomainLevelLabel),
#    (InteractionLevel, InteractionLevelLabel),
#    (SystemLevel, SystemLevelLabel),
# ])


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


@dataclass(frozen=True, kw_only=True)
class Code:
    index: int
    name: str
    tore_index: ToreLabel

    @property
    def level(self) -> ToreLevel:
        for level, level_labels in TORE:
            if self.tore_index in level_labels:
                return cast(ToreLevel, level)

        raise IndexError


@dataclass(frozen=True, kw_only=True)
class Token:
    index: int
    name: str
    lemma: str
    pos: Pos
    tore_codes: List[Code] = dataclasses.field(default_factory=list)

    def __str__(self) -> str:
        return self.name

    @property
    def codes(self) -> List[ToreLabel]:
        return [tc.tore_index for tc in self.tore_codes]

    @property
    def level_codes(self) -> List[ToreLevel]:
        return [tc.level for tc in self.tore_codes]


@dataclass(frozen=True, kw_only=True)
class Sentence:
    tokens: List[Token] = dataclasses.field(default_factory=list)
    source: str

    def __str__(self) -> str:
        return " ".join([str(token) for token in self.tokens])

    def get_label_counts(self) -> Counter[ToreLabel]:
        c: Counter[ToreLabel] = Counter()
        [c.update(t.codes) for t in self.tokens]
        return c

    def get_level_counts(self) -> Counter[ToreLevel]:
        c: Counter[ToreLevel] = Counter()
        [c.update(t.level_codes) for t in self.tokens]
        return c

    def to_dict(self):
        meta = {
            "text": str(self),
            "self": self,
            "source": self.source,
        }
        label = self.get_label_counts()
        level = self.get_level_counts()

        return meta | label | level


@dataclass
class Dataset:
    sentences: List[Sentence] = dataclasses.field(default_factory=list)

    def get_tokens(self):
        return itertools.chain.from_iterable(
            [sentence.tokens for sentence in self.sentences]
        )

    def __add__(self, other: "Dataset") -> "Dataset":
        return Dataset(
            sentences=self.sentences + other.sentences,
        )

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [s.to_dict() for s in self.sentences]
        ).fillna(0)
