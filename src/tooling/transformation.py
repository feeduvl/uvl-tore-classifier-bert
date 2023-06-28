from typing import Optional

import pandas as pd
from tooling.model import Token
from tooling.model import ToreLabel


def transform_token_label(token_label: ToreLabel) -> Optional[ToreLabel]:
    return token_label


def transform_dataset(ds: pd.DataFrame) -> pd.DataFrame:
    ds["tore_label"] = ds["tore_label"].apply(transform_token_label)

    return ds
