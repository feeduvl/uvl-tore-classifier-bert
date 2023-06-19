from classifiers.sner import classify_sentences
from data.sampling import load_split_dataset, TEXT_TEST
import pandas as pd


text_test = load_split_dataset(TEXT_TEST)
classify_sentences(pd.Series(text_test))
