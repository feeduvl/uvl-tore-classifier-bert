from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Dict
from typing import get_args
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from tooling.model import Label
from tooling.model import LABELS_0


@dataclass
class Experiment:
    name: str = MISSING
    random_state: int = 125
    folds: int = 5
    average: str = "macro"
    dataset: str = "prolific"


enum_dict: Dict[str, str] = {k: k for k in LABELS_0}


LABELS_ENUM = Enum("", enum_dict)  # type: ignore[misc]


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


Config = SNERConfig | BiLSTMConfig
