from typing import List, Union, Literal, Tuple, cast, Optional
from data import ToreLabel
from data import Sentence
from dataclasses import dataclass, asdict
import subprocess
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import pandas as pd
from helpers.filehandling import create_file
import itertools
import pickle

from jinja2 import Environment, FileSystemLoader
from pathlib import Path


BASE_PATH = Path(__file__).parent
TEMP_PATH = BASE_PATH.joinpath(Path("./temp"))
RESSOURCES_PATH = BASE_PATH.joinpath(Path("./ressources"))

TRAIN_FILE = Path("./sner_train_file.txt")
CONFIG_FILE = Path("./sner_config_file.prop")
MODEL = Path("./sner.ser.gz")
RESULT = Path("./classification_result.pickle")

TRAIN_FILE_PATH = TEMP_PATH.joinpath(TRAIN_FILE)
CONFIG_FILE_PATH = TEMP_PATH.joinpath(CONFIG_FILE)
MODEL_PATH = TEMP_PATH.joinpath(MODEL)

STANFORD_JAR_PATH = RESSOURCES_PATH.joinpath("./stanford-ner.jar")

CONFIG_TEMPLATE = "ner_training.prop.j2"

CLASSIFICATION_RESULT_PATH = TEMP_PATH.joinpath(RESULT)


@dataclass(frozen=True)
class SNERConfig:
    resultFile: Path = MODEL_PATH.resolve()
    trainFile: Path = TRAIN_FILE_PATH.resolve()


def _sentence_to_token_and_label(sentence: Sentence) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []

    for token in sentence.tokens:
        label: Union[ToreLabel, Literal["0"]]
        try:
            label = token.tore_codes[0].tore_index
        except IndexError:
            label = "0"
        tokens.append(
            (
                token.name,
                label,
            )
        )

    return tokens


def match_tokenization(sentence: Sentence) -> List[Tuple[str, str]]:
    dataset_token_and_labels = _sentence_to_token_and_label(sentence)
    generated_tokens = word_tokenize(str(sentence))

    dataset_char_labels: List[str] = []
    for token, label in dataset_token_and_labels:
        for char in token:
            dataset_char_labels.append(label)

    aligned_tokens: List[Tuple[str, str]] = []
    for token in generated_tokens:
        aligned_tokens.append((token, dataset_char_labels[0]))
        dataset_char_labels = dataset_char_labels[len(token) :]

    return aligned_tokens


def sentences_to_token_df(sentences: pd.Series):
    aligned_tokens = sentences.apply(match_tokenization)

    flattened_aligned_tokens = itertools.chain.from_iterable(
        aligned_tokens.to_list()
    )

    return pd.DataFrame(
        flattened_aligned_tokens,
        columns=["name", "label"],
    )


def create_train_file(sentences: pd.Series):
    def get_labeled_token_for_training(
        sentence: Sentence,
    ) -> List[Tuple[str, str]]:
        tokens = _sentence_to_token_and_label(sentence)
        tokens.append(
            (
                " ",
                " ",
            )
        )
        return tokens

    tokens = sentences.apply(get_labeled_token_for_training)
    labeled_tokens = pd.DataFrame(
        itertools.chain.from_iterable(tokens.to_list()),
        columns=["name", "label"],
    )

    with create_file(TRAIN_FILE_PATH) as tf:
        training_data = "\n".join(
            [
                f"{labeled_token['name']}\t{labeled_token['label']}"
                for index, labeled_token in labeled_tokens.iterrows()
            ]
        )

        tf.write(training_data)
        tf.flush()
    return


def create_config_file(config: SNERConfig = SNERConfig()):
    with create_file(CONFIG_FILE_PATH) as cf:
        template_dir = RESSOURCES_PATH

        environment = Environment(loader=FileSystemLoader(template_dir))
        template = environment.get_template(CONFIG_TEMPLATE)
        content = template.render(**asdict(config))

        cf.write(content)
        cf.flush()
    return


def train_sner() -> None:
    executable = STANFORD_JAR_PATH

    with subprocess.Popen(
        args=[
            "java",
            "-cp",
            executable,
            "edu.stanford.nlp.ie.crf.CRFClassifier",
            "-prop",
            CONFIG_FILE_PATH.resolve(),
            "-Xmx3744M",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    ) as process:
        while process.poll() is None:
            if process.stdout:
                output = process.stdout.readline()
                print(output, end="")


def instantiate_tagger():
    st = StanfordNERTagger(
        model_filename=str(MODEL_PATH.resolve()),
        path_to_jar=str(STANFORD_JAR_PATH),
        encoding="utf-8",
    )
    return st


def classify_sentence(text: str, tagger) -> List[Tuple[str, str]]:
    tokenized_text = word_tokenize(text)
    classified_text = tagger.tag(tokenized_text)

    return [
        (
            classified_token[0],
            classified_token[1],
        )
        for classified_token in classified_text
    ]


def classify_sentences(sentences: pd.Series):
    st = instantiate_tagger()
    # classification_result = sentences.apply(classify_sentence, args=(st,))

    sentence_list = sentences.apply(word_tokenize).to_list()

    classification_result = pd.DataFrame(
        itertools.chain.from_iterable(st.tag_sents(sentence_list)),
        columns=["name", "label"],
    )

    with create_file(
        CLASSIFICATION_RESULT_PATH, mode="wb", encoding=None, buffering=-1
    ) as f:
        f.write(pickle.dumps(classification_result))


def load_classification_result() -> pd.DataFrame:
    with open(CLASSIFICATION_RESULT_PATH, mode="rb") as pickle_file:
        dataset = pickle.load(pickle_file)
    return dataset
