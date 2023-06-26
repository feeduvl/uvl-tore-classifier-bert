# %%
from classifiers.sner import classify_sentences
from classifiers.sner import create_config_file
from classifiers.sner import create_train_file
from classifiers.sner import load_classification_result
from classifiers.sner import sentences_to_token_df
from classifiers.sner import train_sner
from data import DATASETS
from data import FORUM
from data import PROLIFIC
from tooling import evaluation
from tooling.loading import import_dataset
from tooling.loading import load_dataset
from tooling.model import TORE_LABELS
from tooling.sampling import LABELS_TEST
from tooling.sampling import LABELS_TRAIN
from tooling.sampling import load_split_dataset
from tooling.sampling import split_dataset
from tooling.sampling import TEXT_TEST

NAME = "DEFAULT"

import_dataset(name=NAME, ds_spec=FORUM)

# %%

d = load_dataset(name=NAME)

split_dataset(name=NAME, text=d["text"], labels=d["self"], test_size=0.2, stratify=d["source"], random_state=125
)

# %%

labels_train = load_split_dataset(name=NAME, filename=LABELS_TRAIN)

create_config_file(name=NAME)
create_train_file(name=NAME, sentences=labels_train)

train_sner(name=NAME)

# %%

text_test = load_split_dataset(name=NAME, filename=TEXT_TEST)
classify_sentences(name=NAME, sentences=text_test)

# %%

results = load_classification_result(name=NAME)

labels_test = load_split_dataset(name=NAME, filename=LABELS_TEST)
solution = sentences_to_token_df(labels_test)

# %%

p = evaluation.score_precision(solution=solution["label"], results=results['label'], labels=TORE_LABELS)


# %%


r = evaluation.score_recall(solution=solution["label"], results=results['label'], labels=TORE_LABELS)

# %%


conf = evaluation.confusion_matrix(name=NAME, solution=solution["label"], results=results["label"])
