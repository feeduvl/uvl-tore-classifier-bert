from typing import List, Union, Literal
from data import ToreLabel
from data import Sentence
from dataclasses import dataclass, asdict
import subprocess
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import pandas as pd
from helpers.filehandling import create_file
import itertools

from jinja2 import Environment, FileSystemLoader
from pathlib import Path


BASE_PATH = Path(__file__).parent
TEMP_PATH = BASE_PATH.joinpath(Path("./temp"))
RESSOURCES_PATH = BASE_PATH.joinpath(Path("./ressources"))

TRAIN_FILE = Path("./sner_train_file.txt")
CONFIG_FILE = Path("./sner_config_file.prop")
MODEL = Path("./sner.ser.gz")


TRAIN_FILE_PATH = TEMP_PATH.joinpath(TRAIN_FILE)
CONFIG_FILE_PATH = TEMP_PATH.joinpath(CONFIG_FILE)
MODEL_PATH = TEMP_PATH.joinpath(MODEL)

STANFORD_JAR_PATH = RESSOURCES_PATH.joinpath("./stanford-ner.jar")

CONFIG_TEMPLATE = "ner_training.prop.j2"


@dataclass(frozen=True)
class LabeledToken:
    name: str
    label: str


@dataclass(frozen=True)
class SNERConfig:
    resultFile: Path = MODEL_PATH.resolve()
    trainFile: Path = TRAIN_FILE_PATH.resolve()


def _get_token(sentence: Sentence) -> List[LabeledToken]:
    tokens: List[LabeledToken] = []

    for token in sentence.tokens:
        label: Union[ToreLabel, Literal["0"]]
        try:
            label = token.tore_codes[0].tore_index
        except IndexError:
            label = "0"
        tokens.append(LabeledToken(name=token.name, label=label))
    # append empty token to signal sentence end
    tokens.append(LabeledToken(name=" ", label=" "))

    return tokens


def create_train_file(sentences: pd.Series):
    tokens = sentences.apply(_get_token)
    labeled_tokens = itertools.chain.from_iterable(tokens.to_list())

    # labeled_tokens = _get_token(sentences)

    with create_file(TRAIN_FILE_PATH) as tf:
        training_data = "\n".join(
            [
                f"{labeled_token.name}\t{labeled_token.label}"
                for labeled_token in labeled_tokens
            ]
        )

        tf.write(training_data)
        tf.flush()
    return


def create_config_file(config: SNERConfig):
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

    process = subprocess.Popen(
        args=[
            "java",
            "-cp",
            executable,
            "edu.stanford.nlp.ie.crf.CRFClassifier",
            "-prop",
            CONFIG_FILE_PATH.resolve(),
        ],
        universal_newlines=True,
    )

    while process.poll is None:
        output = process.stdout.readline()
        print(output.strip())
        # Do something else
        return_code = process.poll()
        if return_code is not None:
            print("RETURN CODE", return_code)

            # Process has finished, read rest of the output
            for output in process.stdout.readlines():
                print(output.strip())
            break


def classify(self, dataset):
    st = StanfordNERTagger(
        MODEL_PATH,
        STANFORD_JAR_PATH,
        encoding="utf-8",
    )

    tokenized_text = word_tokenize(dataset)
    classified_text = st.tag(tokenized_text)

    return classified_text
