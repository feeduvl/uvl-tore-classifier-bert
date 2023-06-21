from tooling import evaluation

import argparse
from classifiers.sner import (
    sentences_to_token_df,
    load_classification_result,
 
)
import pandas as pd
from tooling import load_split_dataset, LABELS_TEST, TORE_LABELS

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="default")
args = parser.parse_args()
name = args.name


results = load_classification_result(name=name)

labels_test = load_split_dataset(name=name,filename=LABELS_TEST)
solution = sentences_to_token_df(pd.Series(labels_test))


live = evaluation.create_live(name=name)
evaluation.score_precision(live, solution=solution['label'], results=results['label'], labels=TORE_LABELS)
evaluation.score_recall(live, solution=solution['label'], results=results['label'], labels=TORE_LABELS)
evaluation.confusion_matrix(name=name, live=live, solution=solution['label'], results=results['label'] )
evaluation.summarize(live=live)