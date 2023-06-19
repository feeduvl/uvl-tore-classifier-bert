import argparse
import sys


from data import import_dataset, FORUM, PROLIFIC


parser = argparse.ArgumentParser()

parser.add_argument(
    "--dataset", choices=["forum", "prolific"], action="append"
)

args = parser.parse_args()
dataset_list = args.dataset


datasets = []
for d in dataset_list:
    if d == "forum":
        datasets += FORUM
    if d == "prolific":
        datasets += PROLIFIC

# execute

import_dataset(datasets)
