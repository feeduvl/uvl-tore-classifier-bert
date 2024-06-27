from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from itertools import product
from itertools import starmap
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import TypeVar

from omegaconf import MISSING

from tooling.logging import logging_setup
from tooling.model import LABELS_NONE

T = TypeVar("T", bound=Callable[..., Any])


logging = logging_setup(__name__)

enum_dict: Dict[str, str] = {k: k for k in LABELS_NONE}
LABELS_ENUM = Enum("", enum_dict)  # type: ignore[misc]


def get_variants(config_class: T, **items: Dict[str, Any]) -> List[T]:
    # del items["config_class"]
    variants = product(*items.values())

    instances: List[Any] = []
    for variant in list(variants):
        config: Dict[str, Any] = dict(zip(items.keys(), variant, strict=True))
        instance = config_class(**config)
        instances.append(instance)

    return instances


@dataclass
class Experiment:
    name: str = MISSING
    description: str = ""
    random_state: int = 125
    folds: int = 5
    iterations: Optional[int] = None
    average: str = "macro"
    dataset: str = "prolific"
    lower_case: bool = False
    force: bool = True
    pin_commit: bool = False
    smote: bool = False
    smote_k_neighbors: int = 5
    smote_sampling_strategy: str = 'not majority'
    smote_balance_to_average: bool = False

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
    system_level: Optional[str] = None

@dataclass
class SNER:
    type: str = "SNER"


@dataclass
class BiLSTM:
    type: str = "BiLSTM"
    sentence_length: Optional[int] = 106
    batch_size: int = 32
    number_epochs: int = 4
    verbose: int = 1
    weighted_classes: bool = False
    learning_rate: float = 0.0001


@dataclass
class BERT:
    #model: str = "bert-base-uncased"
    model: str = "bert-large-uncased"
    #model: str= "bert-large-uncased-whole-word-masking"
    type: str = "BERT"
    max_len: Optional[int] = 106
    train_batch_size: int = 32
    validation_batch_size: int = 32
    number_epochs: int = 5
    learning_rate_bert: float = 2e-05
    learning_rate_classifier: float = 0.01
    weight_decay: float = 0.01
    weighted_classes: bool = False

@dataclass
class RoBERTa:
    #model: str = "roberta-large"
    model: str = "Jean-Baptiste/roberta-large-ner-english"
    type: str = "RoBERTa"
    max_len: Optional[int] = 106
    train_batch_size: int = 32
    validation_batch_size: int = 32
    number_epochs: int = 5
    learning_rate_roberta: float = 2e-05
    learning_rate_classifier: float = 0.01
    weight_decay: float = 0.01
    weighted_classes: bool = False



@dataclass
class PreTrainedBERT:
    model_path: str


@dataclass
class StagedBERT(BERT):
    layers: List[int] = field(default_factory=list)


@dataclass
class SNERConfig:
    sner: SNER = field(default_factory=SNER)

    experiment: Experiment = field(default_factory=Experiment)
    transformation: Transformation = field(default_factory=Transformation)


@dataclass
class BiLSTMConfig:
    bilstm: BiLSTM = field(default_factory=BiLSTM)

    experiment: Experiment = field(default_factory=Experiment)
    transformation: Transformation = field(default_factory=Transformation)


@dataclass
class BERTConfig:
    bert: BERT = field(default_factory=BERT)

    experiment: Experiment = field(default_factory=Experiment)
    transformation: Transformation = field(default_factory=Transformation)

@dataclass
class RoBERTaConfig:
    roberta: RoBERTa = field(default_factory=RoBERTa)

    experiment: Experiment = field(default_factory=Experiment)
    transformation: Transformation = field(default_factory=Transformation)

@dataclass
class StagedBERTConfig:
    bert: StagedBERT = field(default_factory=StagedBERT)

    experiment: Experiment = field(default_factory=Experiment)
    transformation: Transformation = field(default_factory=Transformation)
    hint_transformation: Transformation = field(default_factory=Transformation)


FirstStageConfigs = SNERConfig | BiLSTMConfig | BERTConfig


@dataclass
class DualModelStagedBERTConfig:
    first_model_bert: Optional[BERTConfig] = None
    first_model_bilstm: Optional[BiLSTMConfig] = None
    first_model_sner: Optional[SNERConfig] = None

    bert: StagedBERT = field(default_factory=StagedBERT)

    experiment: Experiment = field(default_factory=Experiment)
    transformation: Transformation = field(default_factory=Transformation)


Config = FirstStageConfigs | DualModelStagedBERTConfig | StagedBERTConfig
