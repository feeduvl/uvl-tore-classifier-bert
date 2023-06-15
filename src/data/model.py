from pydantic import BaseModel
from datetime import datetime
from typing import List, Literal, Optional, Union, get_args, Tuple, cast
from collections import Counter
import pandas as pd
import itertools
import dataclasses

from pydantic.dataclasses import (
    dataclass,
)


Pos = Literal["v", "n", "a", "r", ""]


DomainLevelLabel = Literal[
    "Task",
    "Goals",
    "Domain Data",
    "Activity",
    "Stakeholder",
]

InteractionLevelLabel = Literal[
    "System Function",
    "Interaction",
    "Interaction Data",
    "Workspace",
]

SystemLevelLabel = Literal[
    "Software",
    "Internal Action",
    "Internal Data",
]

AdditionalLabel = Literal["Relationship"]

ToreLabel = Literal[SystemLevelLabel, InteractionLevelLabel, DomainLevelLabel]

DomainLevel = Literal["Domain Level"]
InteractionLevel = Literal["Interaction Level"]
SystemLevel = Literal["System Level"]
ToreLevel = Literal[DomainLevel, InteractionLevel, SystemLevel]

ToreLevelLabels: List[Tuple[ToreLevel, ToreLabel]] = [
    (DomainLevel, DomainLevelLabel),
    (InteractionLevel, InteractionLevelLabel),
    (SystemLevel, SystemLevelLabel),
]


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
    tore: Union[ToreLabel, Literal[""], AdditionalLabel]


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
        for level, level_labels in ToreLevelLabels:
            if self.tore_index in get_args(level_labels):
                return cast(ToreLevel, get_args(level)[0])

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
