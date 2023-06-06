from pydantic import BaseModel
from datetime import datetime
from typing import List, Literal, ItemsView, TypedDict
from collections import Counter


Pos = Literal["v", "n", "a", "r", ""]


class ImportDoc(BaseModel):
    name: str
    begin_index: int
    end_index: int


class ImportToken(BaseModel):
    index: int
    name: str
    lemma: str
    pos: Pos
    num_name_codes: int
    num_tore_codes: int


class ImportCode(BaseModel):
    index: int
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


class Code(BaseModel):
    index: int
    name: str
    tore_index: str
    tokens: List["Token"] = []


class Token(BaseModel):
    index: int
    name: str
    lemma: str
    pos: Pos
    tore_codes: List[Code] = []

    def __str__(self) -> str:
        return self.name

    @property
    def codes(self) -> List[str]:
        return [tc.tore_index for tc in self.tore_codes]


class Sentence(BaseModel):
    tokens: List[Token] = []

    def __str__(self) -> str:
        return " ".join([str(token) for token in self.tokens])

    def get_label_counts(self) -> Counter[str]:
        c: Counter[str] = Counter()
        [c.update(t.codes) for t in self.tokens]
        return c

    def to_dict(self):
        return {"text": str(self)} | self.get_label_counts()


class Doc(BaseModel):
    name: str
    sentences: List[Sentence] = []
    content: List[Token] = []


class Dataset(BaseModel):
    docs: List[Doc] = []
    tokens: List[Token] = []
    codes: List[Code] = []
    sentences: List[Sentence] = []
