import itertools
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from typing import List
from typing import Tuple

import pandas as pd
from jinja2 import Environment
from jinja2 import FileSystemLoader
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from strictly_typed_pandas.dataset import DataSet

from classifiers.sner.files import configfile
from classifiers.sner.files import modelfile
from classifiers.sner.files import RESSOURCES_PATH
from classifiers.sner.files import resultfile_csv
from classifiers.sner.files import resultfile_pickle
from classifiers.sner.files import solutionfile_csv
from classifiers.sner.files import solutionfile_pickle
from classifiers.sner.files import STANFORD_JAR_PATH
from classifiers.sner.files import TEMPLATE_FILENAME
from classifiers.sner.files import trainfile
from data import create_file
from data import PickleAndCSV
from tooling.logging import logging_setup
from tooling.model import data_to_sentences
from tooling.model import DataDF
from tooling.model import ResultDF
from tooling.model import ToreLabel
from tooling.observability import log_artifacts

logging = logging_setup(__name__)


@dataclass(frozen=True)
class SNERConfig:
    resultFile: Path
    trainFile: Path


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

    log_artifacts(filepath)
    logging.info(f"Created train file for {iteration=}, stored at {filepath=}")

    return filepath


def create_config_file(name: str, iteration: int) -> Path:
    config = SNERConfig(
        resultFile=modelfile(name=name, iteration=iteration),
        trainFile=trainfile(name=name, iteration=iteration),
    )

    filepath = configfile(name=name, iteration=iteration)

    with create_file(filepath) as cf:
        template_dir = RESSOURCES_PATH

        environment = Environment(loader=FileSystemLoader(template_dir))
        template = environment.get_template(TEMPLATE_FILENAME)
        content = template.render(**asdict(config))

        cf.write(content)
        cf.flush()

    log_artifacts(filepath)
    logging.info(
        f"Created config file for {iteration=}, stored at {filepath=}"
    )

    return filepath


# Train


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
        modulo = 10

        logging.info(f"Starting training, only printing every {modulo}. line.")
        line_counter = 0
        while process.poll() is None:
            if process.stdout:
                output = process.stdout.readline()
                line_counter += 1

                if line_counter % modulo == 0:
                    logging.info(output)
                else:
                    logging.debug(output)

    filepath = modelfile(name=name, iteration=iteration)

    log_artifacts(filepath)
    logging.info(f"Trained model for {iteration=}, stored at {filepath=}")

    return filepath


# Classify


def instantiate_tagger(path: Path) -> StanfordNERTagger:
    st = StanfordNERTagger(
        model_filename=str(path),
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
) -> PickleAndCSV:
    path = modelfile(name=name, iteration=iteration)

    classification_result = classify_sentences_action(
        path, data_test=data_test
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

    logging.info(
        f"Logged classification results for iteration {iteration} at {pickle_path}, {csv_path}"
    )
    log_artifacts(csv_path)
    log_artifacts(pickle_path)

    return PickleAndCSV(pickle_file=pickle_path, csv_file=csv_path)


def classify_sentences_action(
    modelfile: Path, data_test: DataSet[DataDF]
) -> DataSet[ResultDF]:
    st = instantiate_tagger(modelfile)

    sentences = data_to_sentences(data=data_test)

    sentence_list = list(map(word_tokenize, sentences))

    classification_result: DataSet[ResultDF] = cast(
        DataSet[ResultDF],
        pd.DataFrame(
            itertools.chain.from_iterable(st.tag_sents(sentence_list)),
            columns=["string", "tore_label"],
        ),
    )
    return classification_result


def realign_results(
    input: DataSet[DataDF], output: DataSet[ResultDF]
) -> DataSet[ResultDF]:
    input.reset_index(drop=True, inplace=True)
    output.reset_index(drop=True, inplace=True)

    for row in input.itertuples():
        index = row[0]
        tokens = word_tokenize(row.string)
        if len(tokens) > 1:
            for i in range(len(tokens) - 1):
                output = cast(
                    DataSet[ResultDF],
                    output.drop([index + 1]).reset_index(drop=True),
                )

    return output


# Prepare solution


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
) -> PickleAndCSV:
    aligned_tokens = match_tokenization(data_test=data_test)

    solution_df = pd.DataFrame(
        aligned_tokens, columns=["string", "tore_label"]
    )

    solution_df.fillna("0", inplace=True)

    csv_path = solutionfile_csv(name=name, iteration=iteration)
    with create_file(
        csv_path,
        mode="wb",
        encoding=None,
        buffering=-1,
    ) as f:
        solution_df.to_csv(f)

    pickle_path = solutionfile_pickle(name=name, iteration=iteration)
    with create_file(
        pickle_path,
        mode="wb",
        encoding=None,
        buffering=-1,
    ) as f:
        solution_df.to_pickle(f)

    logging.info(
        f"Logged solutions for iteration {iteration} at {pickle_path}, {csv_path}"
    )
    log_artifacts(csv_path)
    log_artifacts(pickle_path)

    return PickleAndCSV(pickle_file=pickle_path, csv_file=csv_path)


# Evaluation
