from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict
from typing import Optional

from strictly_typed_pandas.dataset import DataSet

from tooling.model import Label_None
from tooling.model import ResultDF
from tooling.model import ToreLabelDF


@dataclass
class IterationResult:
    step: Optional[int]
    result: DataSet[ResultDF] | DataSet[ToreLabelDF]
    solution: DataSet[ResultDF] | DataSet[ToreLabelDF]

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    pl_precision: Dict[Label_None, float] = field(default_factory=dict)
    pl_recall: Dict[Label_None, float] = field(default_factory=dict)

    confusion_matrix: Optional[Path] = None

    label_count: int = 0


@dataclass
class ExperimentResult:
    label_count: int = 0

    min_f1: float = 0.0
    min_precision: float = 0.0
    min_recall: float = 0.0
    mean_precision: float = 0.0
    mean_f1: float = 0.0
    mean_recall: float = 0.0
    max_precision: float = 0.0
    max_recall: float = 0.0
    max_f1: float = 0.0

    pl_mean_precision: Dict[Label_None, float] = field(default_factory=dict)
    pl_mean_recall: Dict[Label_None, float] = field(default_factory=dict)

    confusion_matrix: Path = Path()
