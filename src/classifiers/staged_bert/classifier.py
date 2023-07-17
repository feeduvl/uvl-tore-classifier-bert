from functools import partial
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import omegaconf

from tooling.config import Transformation
from tooling.model import Label_None_Pad
from tooling.model import LABELS_NONE
from tooling.model import TORE_LABELS
from tooling.model import TORE_LABELS_NONE
from tooling.model import TORE_LEVEL
from tooling.model import ToreLabel
from tooling.model import ToreLevel
from tooling.model import ZERO
from tooling.transformation import transform_token_label


def generate_hint_data(
    tore_label_ids: List[List[int]],
    id2label: Dict[int, Label_None_Pad],
    hint_transformation: partial[Optional[Label_None_Pad]],
    hint_label2id: Dict[Label_None_Pad, int],
) -> List[List[int]]:
    hints = []

    for tore_label_list in tore_label_ids:
        hint_list = []
        for label_id in tore_label_list:
            if label_id == -100:
                hint_list.append(-100)
                continue
            label = id2label[label_id]
            transformed_label = hint_transformation(label)
            if transformed_label is None:
                hint_list.append(hint_label2id[ZERO])
            else:
                hint_list.append(hint_label2id[transformed_label])

        assert len(hint_list) == len(tore_label_list)
        hints.append(hint_list)

    return hints


def get_hint_transformation(
    cfg: Transformation,
) -> Tuple[partial[Optional[Label_None_Pad]], List[Label_None_Pad]]:
    dict_cfg = omegaconf.OmegaConf.to_container(cfg)

    if not isinstance(dict_cfg, dict):
        raise ValueError("No config passed")

    del dict_cfg["description"]
    del dict_cfg["type"]

    hint_labels: List[Label_None_Pad] = ["0"]

    for new_value in dict_cfg.values():
        if new_value is None:
            continue
        elif new_value in LABELS_NONE:
            hint_labels.append(new_value)
            continue
        else:
            raise ValueError(
                f"Transformation value '{new_value}' isn't valid TORE_LABEL"
            )

    return partial(transform_token_label, cfg=dict_cfg), list(set(hint_labels))
