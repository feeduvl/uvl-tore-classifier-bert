import argparse
import sys

from data import FORUM
from data import PROLIFIC
from tooling import import_dataset


parser = argparse.ArgumentParser()

parser.add_argument('--name', default="default")
parser.add_argument(
    "--dataset", choices=["forum", "prolific"], action="append"
)

args = parser.parse_args()
name = args.name
dataset_list = args.dataset


datasets = []
for d in dataset_list:
    if d == "forum":
        datasets += FORUM
    if d == "prolific":
        datasets += PROLIFIC

# execute

import_dataset(name=name, ds_spec=datasets)
