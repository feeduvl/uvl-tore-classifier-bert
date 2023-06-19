import argparse

from data import load_dataset, split_dataset


parser = argparse.ArgumentParser()

parser.add_argument("--test_size", type=float)
parser.add_argument("--random_state", type=int)


args = parser.parse_args()
test_size = args.test_size
random_state = args.random_state


# execute

d = load_dataset()
split_dataset(
    d["text"],
    d["self"],
    test_size=test_size,
    stratify=d["source"],
    random_state=random_state,
)