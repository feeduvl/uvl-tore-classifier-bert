from pathlib import Path

from data import staged_bert_filepath


MODEL_FILENAME = "bert"
OUTPUT_FILENAME = "output"
LOGGING_FILENAME = "logging"


def model_path(name: str, iteration: int) -> Path:
    return staged_bert_filepath(
        name=name,
        filename=f"{iteration}_" + MODEL_FILENAME,
    )


def output_path(name: str, clean: bool) -> Path:
    return staged_bert_filepath(
        name=name, filename=OUTPUT_FILENAME, clean=clean
    )


def logging_path(name: str, clean: bool) -> Path:
    return staged_bert_filepath(
        name=name, filename=LOGGING_FILENAME, clean=clean
    )
