from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Dict
from typing import Optional

from omegaconf import MISSING

from tooling.logging import logging_setup
from tooling.model import LABELS_NONE

logging = logging_setup(__name__)

enum_dict: Dict[str, str] = {k: k for k in LABELS_NONE}
LABELS_ENUM = Enum("", enum_dict)  # type: ignore[misc]


@dataclass
class Experiment:
    name: str = MISSING
    random_state: int = 125
    folds: int = 5
    iterations: Optional[int] = None
    average: str = "macro"
    dataset: str = "prolific"
    lower_case: bool = False


@dataclass
class Transformation:
    description: str = MISSING
    type: str = MISSING

    task: Optional[str] = None
    goals: Optional[str] = None
    domain_data: Optional[str] = None
    activity: Optional[str] = None
    stakeholder: Optional[str] = None

    system_function: Optional[str] = None
    interaction: Optional[str] = None
    interaction_data: Optional[str] = None
    workspace: Optional[str] = None

    software: Optional[str] = None
    internal_action: Optional[str] = None
    internal_data: Optional[str] = None


@dataclass
class Meta:
    mlflow_tracking_uri: str = "https://bockstaller.cc"


@dataclass
class SNER:
    type: str = "SNER"


@dataclass
class BiLSTM:
    type: str = "BiLSTM"
    sentence_length: Optional[int] = None
    batch_size: int = 32
    number_epochs: int = 32
    validation_split: float = 0.1
    verbose: int = 1


@dataclass
class BERT:
    model: str = "bert-base-cased"
    type: str = "BERT"
    max_len: Optional[int] = 106
    train_batch_size: int = 32
    validation_batch_size: int = 32
    number_epochs: int = 32
    learning_rate: float = 2e-05
    weight_decay: float = 0.01
    weighted_classes: bool = False


@dataclass
class SNERConfig:
    sner: SNER = field(default_factory=SNER)
    meta: Meta = field(default_factory=Meta)
    experiment: Experiment = field(default_factory=Experiment)
    transformation: Transformation = field(default_factory=Transformation)


@dataclass
class BiLSTMConfig:
    bilstm: BiLSTM = field(default_factory=BiLSTM)
    meta: Meta = field(default_factory=Meta)
    experiment: Experiment = field(default_factory=Experiment)
    transformation: Transformation = field(default_factory=Transformation)


@dataclass
class BERTConfig:
    bert: BERT = field(default_factory=BERT)
    meta: Meta = field(default_factory=Meta)
    experiment: Experiment = field(default_factory=Experiment)
    transformation: Transformation = field(default_factory=Transformation)


@dataclass
class StagedBERTConfig:
    bert: BERT = field(default_factory=BERT)
    meta: Meta = field(default_factory=Meta)
    experiment: Experiment = field(default_factory=Experiment)
    transformation: Transformation = field(default_factory=Transformation)
    hint_transformation: Transformation = field(default_factory=Transformation)


Config = SNERConfig | BiLSTMConfig | BERTConfig | StagedBERTConfig
