from tooling.sampling import load_split_dataset, LABELS_TRAIN
from classifiers.sner import (
    create_train_file,
    create_config_file,
    train_sner,
)
import pandas as pd
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--name', default="default")

args = parser.parse_args()
name = args.name


# execute
labels_train = load_split_dataset(name=name, filename=LABELS_TRAIN)
create_config_file(name=name)
create_train_file(name=name, sentences=pd.Series(labels_train))

train_sner(name=name)
