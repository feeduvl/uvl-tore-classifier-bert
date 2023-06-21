from classifiers.sner import classify_sentences
from tooling.sampling import load_split_dataset, TEXT_TEST
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="default")
args = parser.parse_args()
name = args.name


text_test = load_split_dataset(name=name, filename=TEXT_TEST)
classify_sentences(name=name, sentences=pd.Series(text_test))
