import itertools
import pickle
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
from data import create_file
from data import sner_filepath
from data import SNER_TEMP
from jinja2 import Environment
from jinja2 import FileSystemLoader
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from strictly_typed_pandas import DataSet
from tooling.model import data_to_sentences
from tooling.model import DataDF
from tooling.model import ResultDF
from tooling.model import ToreLabel

BASE_PATH = Path(__file__).parent
RESSOURCES_PATH = BASE_PATH.joinpath(Path("./ressources"))
STANFORD_JAR_PATH = RESSOURCES_PATH.joinpath("./stanford-ner.jar")


TEMPLATE_FILENAME = "ner_training.prop.j2"


TRAIN_FILENAME = "sner_train_file"
CONFIG_FILENAME = "sner_config_file"
MODEL_FILENAME = "sner"
RESULT_FILENAME = "classification_result"
SOLUTION_FILENAME = "solution"


def trainfile(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name,
        filename=f"{iteration}_" + TRAIN_FILENAME + ".txt",
    )


def configfile(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name,
        filename=f"{iteration}_" + CONFIG_FILENAME + ".prop",
    )


def modelfile(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name,
        filename=f"{iteration}_" + MODEL_FILENAME + ".ser.gz",
    )


def resultfile_csv(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name,
        filename=f"{iteration}_" + RESULT_FILENAME + ".csv",
    )


def resultfile_pickle(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name, filename=f"{iteration}_" + RESULT_FILENAME + ".pickle"
    )


def solutionfile_pickle(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name,
        filename=f"{iteration}_" + SOLUTION_FILENAME + ".pickle",
    )


def solutionfile_csv(name: str, iteration: int) -> Path:
    return sner_filepath(
        name=name,
        filename=f"{iteration}_" + SOLUTION_FILENAME + ".csv",
    )


@dataclass(frozen=True)
class SNERConfig:
    resultFile: Path
    trainFile: Path


def match_tokenization(
    data_test: DataSet[DataDF],
) -> List[Tuple[str, ToreLabel | None]]:
    sentences = data_to_sentences(data=data_test)
    # for every char in a sentence we note down the tore_label
    sentence_tore_masks: List[List[ToreLabel | None]] = []
    for sentence_idx, data in data_test.groupby("sentence_id"):
        sentence_tore_mask: List[ToreLabel | None] = []
        for idx, word in data.iterrows():
            for char in word["string"]:
                sentence_tore_mask.append(word["tore_label"])

        sentence_tore_masks.append(sentence_tore_mask)

    aligned_tokens: List[Tuple[str, ToreLabel | None]] = []
    for sentence, tore_mask in zip(
        sentences, sentence_tore_masks, strict=True
    ):
        # we retrieve the generated tokenization
        generated_tokens = word_tokenize(sentence)
        for generated_token in generated_tokens:
            aligned_tokens.append(
                (
                    generated_token,
                    tore_mask[0],
                )
            )
            tore_mask = tore_mask[len(generated_token) :]

    return aligned_tokens


def create_solution(
    name: str, iteration: int, data_test: DataSet[DataDF]
) -> Tuple[Path, Path]:
    aligned_tokens = match_tokenization(data_test=data_test)

    solution_df = pd.DataFrame(
        aligned_tokens, columns=["string", "tore_label"]
    )

    solution_df.fillna("0", inplace=True)

    csv_filepath = solutionfile_csv(name=name, iteration=iteration)
    with create_file(
        csv_filepath,
        mode="wb",
        encoding=None,
        buffering=-1,
    ) as f:
        solution_df.to_csv(f)

    pickle_filepath = solutionfile_pickle(name=name, iteration=iteration)
    with create_file(
        pickle_filepath,
        mode="wb",
        encoding=None,
        buffering=-1,
    ) as f:
        solution_df.to_pickle(f)

    return (
        csv_filepath,
        pickle_filepath,
    )


def load_solution(name: str, iteration: int) -> DataSet[ResultDF]:
    with open(
        solutionfile_pickle(name=name, iteration=iteration), mode="rb"
    ) as pickle_file:
        dataset = pickle.load(pickle_file)

    return cast(DataSet[ResultDF], dataset)


def create_train_file(
    name: str, data_train: DataSet[DataDF], iteration: int
) -> Path:
    sentence_lengths = data_train.groupby("sentence_id")

    filepath = trainfile(name=name, iteration=iteration)

    with create_file(filepath) as tf:
        lines: List[Tuple[str, str]] = []

        for sentence_idx, data in sentence_lengths:
            for index, row in data.iterrows():
                string = row["string"]

                if row["tore_label"] is None:
                    tore_label = "0"
                else:
                    tore_label = row["tore_label"]
                lines.append((string, tore_label))
            lines.append(("", ""))

        training_data = "\n".join(
            [f"{string}\t{tore_label}" for string, tore_label in lines]
        )

        tf.write(training_data)
        tf.flush()
    return filepath


def create_config_file(name: str, iteration: int) -> Path:
    paths: List[Path] = []

    config = SNERConfig(
        resultFile=modelfile(name=name, iteration=iteration),
        trainFile=trainfile(name=name, iteration=iteration),
    )

    filename = configfile(name=name, iteration=iteration)

    with create_file(filename) as cf:
        template_dir = RESSOURCES_PATH

        environment = Environment(loader=FileSystemLoader(template_dir))
        template = environment.get_template(TEMPLATE_FILENAME)
        content = template.render(**asdict(config))

        cf.write(content)
        cf.flush()
    return filename


def train_sner(name: str, iteration: int) -> Path:
    executable = STANFORD_JAR_PATH

    with subprocess.Popen(
        args=[
            "java",
            "-Xmx4G",
            "-cp",
            executable,
            "edu.stanford.nlp.ie.crf.CRFClassifier",
            "-prop",
            configfile(name=name, iteration=iteration).resolve(),
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

    return modelfile(name=name, iteration=iteration)


def instantiate_tagger(name: str, iteration: int) -> StanfordNERTagger:
    st = StanfordNERTagger(
        model_filename=str(modelfile(name=name, iteration=iteration)),
        path_to_jar=str(STANFORD_JAR_PATH),
        encoding="utf-8",
    )
    return st


def classify_sentence(
    text: str, tagger: StanfordNERTagger
) -> List[Tuple[str, str]]:
    tokenized_text = word_tokenize(text)
    classified_text = tagger.tag(tokenized_text)

    return [
        (
            classified_token[0],
            classified_token[1],
        )
        for classified_token in classified_text
    ]


def classify_sentences(
    name: str, iteration: int, data_test: DataSet[DataDF]
) -> Tuple[Path, Path]:
    st = instantiate_tagger(name=name, iteration=iteration)

    sentences = data_to_sentences(data=data_test)

    sentence_list = list(map(word_tokenize, sentences))

    classification_result: DataSet[ResultDF] = cast(
        DataSet[ResultDF],
        pd.DataFrame(
            itertools.chain.from_iterable(st.tag_sents(sentence_list)),
            columns=["string", "tore_label"],
        ),
    )

    pickle_path = resultfile_pickle(name=name, iteration=iteration)
    with create_file(
        pickle_path,
        mode="wb",
        encoding=None,
        buffering=-1,
    ) as f:
        classification_result.to_pickle(f)

    csv_path = resultfile_csv(name=name, iteration=iteration)
    with create_file(
        csv_path,
        mode="w",
        encoding=None,
        buffering=-1,
    ) as f:
        classification_result.to_csv(f)

    return (
        pickle_path,
        csv_path,
    )


def load_classification_result(name: str, iteration: int) -> pd.DataFrame:
    with open(
        resultfile_pickle(name=name, iteration=iteration), mode="rb"
    ) as pickle_file:
        dataset = pickle.load(pickle_file)
    return cast(pd.DataFrame, dataset)
