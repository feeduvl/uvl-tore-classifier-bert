from pathlib import Path

from data import roberta_filepath

MODEL_FILENAME = "model"
OUTPUT_FILENAME = "output"
LOGGING_FILENAME = "logging"


def model_path(name: str, iteration: int) -> Path:
    return roberta_filepath(
        name=name,
        filename=f"{iteration}_" + MODEL_FILENAME,
    )


def output_path(name: str, clean: bool) -> Path:
    return roberta_filepath(name=name, filename=OUTPUT_FILENAME, clean=clean)


def logging_path(name: str, clean: bool) -> Path:
    return roberta_filepath(name=name, filename=LOGGING_FILENAME, clean=clean)
