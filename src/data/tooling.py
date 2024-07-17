import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any
from typing import IO
from typing import List
from typing import TypedDict

from tooling.logging import logging_setup

logging = logging_setup(__name__)
DATA_ROOT = Path(__file__).parent


# Datasets
IMPORT_PATH = DATA_ROOT.joinpath(Path("./datasets"))
datasets = [
    "forum/anno_test.json",
    "forum/anno_train.json",
    "prolific/TORE_Coded_Answers_1_33.json",
    "prolific/TORE_Coded_Answers_34_66.json",
    "prolific/TORE_Coded_Answers_67_100.json",
    "app/Komoot_AppReview.json",
    
    "relevance/Prolific_1_33_Complete_Relevance_Mapping.json",
    "relevance/Prolific_34_66_Complete_Relevance_Mapping.json",
    "relevance/Prolific_67_100_Complete_Relevance_Mapping.json",
    "relevance/Komoot_AppReview_Relevance_Mapping.json",
    
    "spellchecked_ProlificAndKomootAppReview/Spell_Checked_TORE_Coded_Answers_1_33.json",
    "spellchecked_ProlificAndKomootAppReview/Spell_Checked_TORE_Coded_Answers_34_66.json",
    "spellchecked_ProlificAndKomootAppReview/Spell_Checked_TORE_Coded_Answers_67_100.json",
    "spellchecked_ProlificAndKomootAppReview/Spell_Checked_Komoot_AppReview.json",
    
    "spellchecked_Relevance/Spell_Checked_Prolific_1_33_Complete_Relevance_Mapping.json",
    "spellchecked_Relevance/Spell_Checked_Prolific_34_66_Complete_Relevance_Mapping.json",
    "spellchecked_Relevance/Spell_Checked_Prolific_67_100_Complete_Relevance_Mapping.json",
    "spellchecked_Relevance/Spell_Checked_Komoot_AppReview_Relevance_Mapping.json",
]

dataset_source = ["anno", "anno", "prolific", "prolific", "prolific", "komoot",
                  "prolific_relevance", "prolific_relevance", "prolific_relevance", "komoot_relevance",
                  "spellchecked_prolific", "spellchecked_prolific", "spellchecked_prolific", "spellchecked_komoot", 
                  "spellchecked_prolific_relevance", "spellchecked_prolific_relevance", 
                  "spellchecked_prolific_relevance", "spellchecked_komoot_relevance",]

IMPORT_DATASET_PATHS = [
    IMPORT_PATH.joinpath(Path(dataset)) for dataset in datasets
]
DATASETS = list(zip(dataset_source, IMPORT_DATASET_PATHS))
FORUM = DATASETS[:2]
PROLIFIC = DATASETS[2:5]
KOMOOT = [DATASETS[5]]
KOMOOT_AND_PROLIFIC = DATASETS[2:6]
RELEVANCE = DATASETS[6:10]
SPELLCHECKED = DATASETS[10:14]
SPELLCHECKED_RELEVANCE = DATASETS[14:18]
PAF = [DATASETS[x] for x in [2,3,4,0,1]]
PAK = [DATASETS[x] for x in [2,3,4,5]]
FAK = [DATASETS[x] for x in [0,1,5]]
FAP = [DATASETS[x] for x in [0,1,2,3,4]]
KAP = [DATASETS[x] for x in [5,2,3,4]]
KAF = [DATASETS[x] for x in [5,0,1]]
PFTrainKTest = [DATASETS[x] for x in [2,3,4,0,1,5]]
KFTrainPTest = [DATASETS[x] for x in [5,0,1,2,3,4]]
PKTrainFTest = [DATASETS[x] for x in [2,3,4,5,0,1]]




def get_dataset_information(name: str) -> List[tuple[str, Path]]:
    if name == "forum":
        return FORUM
    if name == "prolific":
        return PROLIFIC
    if name == "komoot":
        return KOMOOT
    if name == "komoot_and_prolific":
        return KOMOOT_AND_PROLIFIC
    if name == "relevance":
        return RELEVANCE
    if name == "spellchecked":
        return SPELLCHECKED
    if name == "spellchecked_relevance":
        return SPELLCHECKED_RELEVANCE
    if name == "all":
        return DATASETS
    if name == "paf":
        return PAF
    if name == "pak":
        return PAK
    if name == "fak":
        return FAK
    if name == "fap":
        return FAP
    if name == "kap":
        return KAP
    if name == "kaf":
        return KAF
    if name == "PFTrainKTest":
        return PFTrainKTest
    if name == "KFTrainPTest":
        return KFTrainPTest
    if name == "PKTrainFTest":
        return PKTrainFTest

    raise ValueError("No dataset with this name available")


def filename(
    basepath: Path, name: str, filename: str, clean: bool = False
) -> Path:
    path = basepath.joinpath(Path(name)).joinpath(Path(filename))

    created_files.append(path)
    if clean:
        shutil.rmtree(path=path, ignore_errors=True)

    path.parent.mkdir(parents=True, exist_ok=True)

    logging.debug(f"Created path: {path}")

    return path


created_files: List[Path] = []


@contextmanager
def create_file(
    file_path: Path,
    mode: str = "w+",
    encoding: str | None = "utf-8",
    buffering: int = 1,
) -> Iterator[IO[Any]]:
    if file_path.exists():
        file_path.unlink()

    created_files.append(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    f = open(file_path, mode=mode, encoding=encoding, buffering=buffering)

    logging.debug(f"Created path: {file_path}")

    try:
        yield f
    finally:
        f.close()


def cleanup_files() -> None:
    for file_path in created_files:
        if file_path.exists():
            if file_path.is_file():
                file_path.unlink()
            else:
                shutil.rmtree(file_path)
        logging.debug(f"Deleted path: {file_path}")


class PickleAndCSV(TypedDict):
    pickle_file: Path
    csv_file: Path


# Temporary directories
TEMP = DATA_ROOT.joinpath(Path("temp"))
LOADING_TEMP = TEMP.joinpath(Path("loading"))
SAMPLING_TEMP = TEMP.joinpath(Path("sampling"))
EVALUATION_TEMP = TEMP.joinpath(Path("evaluation"))
SNER_TEMP = TEMP.joinpath(Path("sner"))
BILSTM_TEMP = TEMP.joinpath(Path("bilstm"))
BERT_TEMP = TEMP.joinpath(Path("bert"))
ROBERTA_TEMP = TEMP.joinpath(Path("roberta"))
STAGED_BERT_TEMP = TEMP.joinpath(Path("staged_bert"))

loading_filepath = partial(filename, basepath=LOADING_TEMP)
sampling_filepath = partial(filename, basepath=SAMPLING_TEMP)
sner_filepath = partial(filename, basepath=SNER_TEMP)
bilstm_filepath = partial(filename, basepath=BILSTM_TEMP)
bert_filepath = partial(filename, basepath=BERT_TEMP)
roberta_filepath = partial(filename, basepath=ROBERTA_TEMP)
staged_bert_filepath = partial(filename, basepath=STAGED_BERT_TEMP)
evaluation_filepath = partial(filename, basepath=EVALUATION_TEMP)
