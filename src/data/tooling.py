from contextlib import contextmanager
from functools import partial
from pathlib import Path


DATA_ROOT = Path(__file__).parent


# Datasets
IMPORT_PATH = DATA_ROOT.joinpath(Path("./datasets"))
datasets = [
    "forum/anno_test.json",
    "forum/anno_train.json",
    "prolific/TORE_Coded_Answers_1_33.json",
    "prolific/TORE_Coded_Answers_34_66.json",
    "prolific/TORE_Coded_Answers_67_100.json",
]
dataset_source = ["anno", "anno", "prolific", "prolific", "prolific"]
IMPORT_DATASET_PATHS = [
    IMPORT_PATH.joinpath(Path(dataset)) for dataset in datasets
]
DATASETS = list(zip(dataset_source, IMPORT_DATASET_PATHS))
FORUM = DATASETS[:2]
PROLIFIC = DATASETS[2:]


# Temporary directories
TEMP = DATA_ROOT.joinpath(Path("temp"))

LOADING_TEMP = TEMP.joinpath(Path("loading"))

SAMPLING_TEMP = TEMP.joinpath(Path("sampling"))

SNER_TEMP = TEMP.joinpath(Path("sner"))
EVALUATION_TEMP = TEMP.joinpath(Path("evaluation"))


def filename(basepath: Path, name: str, filename: str) -> Path:
    path = basepath.joinpath(Path(name)).joinpath(Path(filename))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


loading_filepath = partial(filename, basepath=LOADING_TEMP)
sampling_filepath = partial(filename, basepath=SAMPLING_TEMP)
sner_filepath = partial(filename, basepath=SNER_TEMP)

evaluation_filepath = partial(filename, basepath=EVALUATION_TEMP)


@contextmanager
def create_file(file_path: Path, mode="w+", encoding="utf-8", buffering=1):
    if file_path.exists():
        file_path.unlink()

    file_path.parent.mkdir(parents=True, exist_ok=True)

    f = open(file_path, mode=mode, encoding=encoding, buffering=buffering)
    try:
        yield f
    finally:
        f.close()
