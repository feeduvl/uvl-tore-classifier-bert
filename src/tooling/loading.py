import pickle
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from typing import get_args
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
from data import create_file
from data import loading_filepath
from data import LOADING_TEMP
from pydantic import ValidationError

from .model import ImportCode
from .model import ImportDataSet
from .model import ImportToreLabel
from .model import Token
from .model import ToreLabel

IMPORTED_DATASET_FILENAME_CSV = "imported_dataset.csv"
IMPORTED_DATASET_FILENAME_PICKLE = "imported_dataset.pickle"


def split_tokenlist_into_sentences(tokens: List[Token]) -> List[Token]:
    # split content into sentences

    punctuation = [".", "!", "?"]

    starts: List[int] = [0]
    ends: List[int] = [len(tokens)]
    shift_reg: List[str] = [" ", " ", " "]

    for idx, token in enumerate(tokens):
        # Handle sentence terminator
        shift_reg.insert(0, token.string)
        shift_reg.pop()

        if "".join(shift_reg) == "###":
            ends.append(idx - 2)
            starts.append(idx + 1)

        # Handle punctuation
        if token.string in punctuation:
            try:
                if tokens[idx + 1].string not in punctuation:
                    starts.append(idx + 1)
                    ends.append(idx + 1)
            except IndexError:
                pass

    starts.sort()
    ends.sort()

    result_tokens: List[Token] = []
    for start, end in zip(starts, ends):
        if end - start != 0:
            sentence_uuid = uuid.uuid4()
            sentence = tokens[start:end]
            for idx, token in enumerate(sentence):
                token.sentence_idx = idx
                token.sentence_id = sentence_uuid
            result_tokens += sentence

    return result_tokens


def clean_token(token_str: str) -> Optional[str]:
    cleaned = token_str.replace("\\", "")

    if cleaned == "":
        return None

    return cleaned


@dataclass(frozen=True)
class Code:
    index: int
    name: str
    tore_index: ToreLabel


def denormalize_dataset(
    imported_dataset: ImportDataSet, dataset_source=str
) -> pd.DataFrame:
    tokenindex_codes: dict[int, List[Code]] = {}

    code_skip_set = set()

    imported_code: ImportCode

    for imported_code in imported_dataset.codes:
        # discard empty codes in the dataset
        if imported_code.index is not None:
            # imported_code.tore can be a ToreLabel or Literal["Relationship", ""].
            # We don't want the second kind and check for it.
            # They are added to the code_skip_set to be skipped in the per document loop
            if imported_code.tore in get_args(ImportToreLabel):
                try:
                    code = Code(
                        index=imported_code.index,
                        name=imported_code.name,
                        tore_index=cast(
                            ToreLabel, imported_code.tore.replace(" ", "_")
                        ),
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

    dataset: List[Token] = []
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
                    token_str = clean_token(imported_token.name)
                    if token_str is None:
                        continue

                    try:
                        tore_labels = [t.tore_index for t in tore_codes]
                        tore_label = tore_labels[0]

                    except IndexError:
                        tore_label = None

                    token = Token(
                        string=imported_token.name.replace("\\", ""),
                        lemma=imported_token.lemma,
                        pos=imported_token.pos,
                        source=dataset_source,
                        sentence_id=None,
                        sentence_idx=None,
                        tore_label=tore_label,
                    )
                    tokens.append(token)

        dataset += split_tokenlist_into_sentences(tokens=tokens)

    df = pd.DataFrame(dataset)

    return df


def _import_dataset(dataset_info: Tuple[str, Path]):
    dataset_source, import_path = dataset_info
    print(f"Importing dataset: {dataset_source} from {import_path}")

    imported_ds = ImportDataSet.parse_file(import_path.resolve())
    denormalized_ds = denormalize_dataset(
        imported_dataset=imported_ds, dataset_source=dataset_source
    )

    return denormalized_ds


def import_dataset(name: str, ds_spec: List[Tuple[str, Path]]) -> Path:
    dataframes: List[pd.DataFrame] = []
    for d_spec in ds_spec:
        dataframes.append(_import_dataset(d_spec))

    ds_df = pd.concat(dataframes, ignore_index=True)

    filepath_pickle = loading_filepath(
        name=name, filename=IMPORTED_DATASET_FILENAME_PICKLE
    )

    with create_file(
        file_path=filepath_pickle,
        mode="wb",
        encoding=None,
        buffering=-1,
    ) as f:
        ds_df.to_pickle(f)

    filepath_csv = loading_filepath(
        name=name, filename=IMPORTED_DATASET_FILENAME_CSV
    )

    with create_file(
        file_path=filepath_csv,
        mode="wb",
        encoding=None,
        buffering=-1,
    ) as f:
        ds_df.to_csv(f)

    return filepath_pickle, filepath_csv


def load_dataset(name: str) -> pd.DataFrame:
    filepath = loading_filepath(
        name=name, filename=IMPORTED_DATASET_FILENAME_PICKLE
    )
    with open(
        filepath,
        mode="rb",
    ) as pickle_file:
        dataset = pickle.load(pickle_file)
    return dataset
