import argparse

import pandas as pd
from classifiers.sner import classify_sentences
from tooling.sampling import load_split_dataset
from tooling.sampling import TEXT_TEST

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="default")
args = parser.parse_args()
name = args.name


text_test = load_split_dataset(name=name, filename=TEXT_TEST)
classify_sentences(name=name, sentences=pd.Series(text_test))
