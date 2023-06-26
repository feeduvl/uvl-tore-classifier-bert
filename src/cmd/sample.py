import argparse

from tooling import load_dataset
from tooling import split_dataset


parser = argparse.ArgumentParser()

parser.add_argument("--name", default="default")
parser.add_argument("--test_size", type=float)
parser.add_argument("--random_state", type=int)


args = parser.parse_args()
name = args.name
test_size = args.test_size
random_state = args.random_state


# execute
d = load_dataset(name=name)
split_dataset(
    name=name,
    text=d["text"],
    labels=d["self"],
    test_size=test_size,
    stratify=d["source"],
    random_state=random_state,
)
