from data.sampling import load_split_dataset, LABELS_TRAIN
from classifiers.sner import (
    create_train_file,
    create_config_file,
    train_sner,
)
import pandas as pd


# execute
labels_train = load_split_dataset(LABELS_TRAIN)
create_config_file()
create_train_file(pd.Series(labels_train))

train_sner()
