from pydantic import BaseModel
from datetime import datetime
from typing import List, Literal, Optional
from collections import Counter
import pandas as pd
import itertools
import dataclasses

from pydantic.dataclasses import (
    dataclass,
)


Pos = Literal["v", "n", "a", "r", ""]


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
    tore: str


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
    tore_index: str


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
    def codes(self) -> List[str]:
        return [tc.tore_index for tc in self.tore_codes]


@dataclass(frozen=True, kw_only=True)
class Sentence:
    tokens: List[Token] = dataclasses.field(default_factory=list)

    def __str__(self) -> str:
        return " ".join([str(token) for token in self.tokens])

    def get_label_counts(self) -> Counter[str]:
        c: Counter[str] = Counter()
        [c.update(t.codes) for t in self.tokens]
        return c

    def to_dict(self):
        return {"text": str(self)} | self.get_label_counts()


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
        return pd.DataFrame.from_records([s.to_dict() for s in self.sentences])
