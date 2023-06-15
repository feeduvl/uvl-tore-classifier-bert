from typing import List, Tuple, cast, get_args
from pydantic import ValidationError
from .model import (
    ImportDataSet,
    Token,
    Code,
    Sentence,
    Dataset,
    ToreLabel,
    ImportCode,
)
from pathlib import Path
import pickle

from helpers.filehandling import create_file


BASE_PATH = Path(__file__).parent
IMPORT_PATH = BASE_PATH.joinpath(Path("./data"))

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
TEMP_PATH = BASE_PATH.joinpath(Path("./temp"))

LOADED_DATASET_PATHS = [
    TEMP_PATH.joinpath(Path(dataset)).with_suffix(".pickle")
    for dataset in datasets
]
DATASETS = list(
    zip(dataset_source, IMPORT_DATASET_PATHS, LOADED_DATASET_PATHS)
)
FORUM = DATASETS[:2]
PROLIFIC = DATASETS[2:]


def split_tokenlist_into_sentences(
    tokens: List[Token], source: str
) -> List[Sentence]:
    # split content into sentences

    punctuation = [".", "!", "?"]

    starts: List[int] = [0]
    ends: List[int] = [len(tokens)]
    shift_reg: List[str] = [" ", " ", " "]

    for idx, token in enumerate(tokens):
        # Handle sentence terminator
        shift_reg.insert(0, token.name)
        shift_reg.pop()

        if "".join(shift_reg) == "###":
            ends.append(idx - 2)
            starts.append(idx + 1)

        # Handle punctuation
        if token.name in punctuation:
            try:
                if tokens[idx + 1].name not in punctuation:
                    starts.append(idx + 1)
                    ends.append(idx + 1)
            except IndexError:
                pass

    starts.sort()
    ends.sort()

    doc_sentences: List[Sentence] = []
    for start, end in zip(starts, ends):
        if end - start != 0:
            s = Sentence(tokens=[t for t in tokens[start:end]], source=source)
            doc_sentences.append(s)

    return doc_sentences


def denormalize_dataset(
    imported_dataset: ImportDataSet, dataset_source=str
) -> Dataset:
    tokenindex_codes: dict[int, List[Code]] = {}

    code_skip_set = set()

    imported_code: ImportCode

    for imported_code in imported_dataset.codes:
        # discard empty codes in the dataset
        if imported_code.index is not None:
            # imported_code.tore can be a ToreLabel or Literal["Relationship", ""].
            # We don't want the second kind and check for it.
            # They are added to the code_skip_set to be skipped in the per document loop

            if imported_code.tore in get_args(ToreLabel):
                try:
                    code = Code(
                        index=imported_code.index,
                        name=imported_code.name,
                        tore_index=cast(ToreLabel, imported_code.tore),
                    )
                    for token_id in imported_code.tokens:
                        try:
                            tokenindex_codes[token_id]
                        except KeyError:
                            tokenindex_codes[token_id] = []

                        tokenindex_codes[token_id].append(code)
                except ValidationError as e:
                    print(e)
            else:
                for token_id in imported_code.tokens:
                    code_skip_set.add(token_id)

    sentences: List[Sentence] = []
    for imported_document in imported_dataset.docs:
        tokens: List[Token] = []
        for token_index in range(
            imported_document.begin_index, imported_document.end_index
        ):
            imported_token = imported_dataset.tokens[token_index]
            if (imported_token.index is not None) and (
                imported_token.index not in code_skip_set
            ):
                try:
                    tore_codes = tokenindex_codes[imported_token.index]

                except KeyError as e:
                    if imported_token.num_tore_codes == 0:
                        tore_codes = []
                    else:
                        print(imported_token)
                        raise e
                finally:
                    token = Token(
                        index=imported_token.index,
                        name=imported_token.name,
                        lemma=imported_token.lemma,
                        pos=imported_token.pos,
                        tore_codes=tore_codes,
                    )
                    tokens.append(token)

        new_sentences = split_tokenlist_into_sentences(
            tokens=tokens, source=dataset_source
        )
        sentences += new_sentences

    return Dataset(sentences=sentences)


def _import_dataset(dataset_info: Tuple[str, Path, Path]):
    dataset_source, import_path, pickle_path = dataset_info
    print(dataset_source)
    imported_ds = ImportDataSet.parse_file(import_path.resolve())
    denormalized_ds = denormalize_dataset(
        imported_dataset=imported_ds, dataset_source=dataset_source
    )
    with create_file(pickle_path, mode="wb", encoding=None, buffering=-1) as f:
        f.write(pickle.dumps(denormalized_ds))

    return denormalized_ds


def _load_dataset(dataset_info: Tuple[str, Path, Path]):
    dataset_source, import_path, pickle_path = dataset_info
    with open(pickle_path, mode="rb") as pickle_file:
        dataset = pickle.load(pickle_file)
    return dataset


def import_dataset(ds_spec: List[Tuple[str, Path, Path]]):
    for d_spec in ds_spec:
        _import_dataset(d_spec)


def load_dataset(ds_spec: List[Tuple[str, Path, Path]]):
    ds: Dataset = Dataset()
    for dataset in ds_spec:
        ds += _load_dataset(dataset)
    return ds
