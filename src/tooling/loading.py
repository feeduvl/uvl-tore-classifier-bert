from typing import List, Tuple, cast, get_args, Optional
from pydantic import ValidationError
from .model import (
    ImportDataSet,
    Token,
    Code,
    Sentence,
    Dataset,
    ImportToreLabel,
    ToreLabel,
    ImportCode,
)
from pathlib import Path
import pickle


from data import LOADING_TEMP, create_file, loading_filepath

IMPORTED_DATASET_FILENAME = "imported_dataset.pickle"


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


def clean_token(token_str: str) -> Optional[str]:
    cleaned = token_str.replace("\\", "")

    if cleaned == "":
        return None

    return cleaned


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
                    token_str = clean_token(imported_token.name)
                    if token_str is None:
                        continue

                    token = Token(
                        index=imported_token.index,
                        name=imported_token.name.replace("\\", ""),
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


def _import_dataset(dataset_info: Tuple[str, Path]):
    dataset_source, import_path = dataset_info
    print(f"Importing dataset: {dataset_source} from {import_path}")

    

    imported_ds = ImportDataSet.parse_file(import_path.resolve())
    denormalized_ds = denormalize_dataset(
        imported_dataset=imported_ds, dataset_source=dataset_source
    )

    return denormalized_ds





def import_dataset(name: str, ds_spec: List[Tuple[str, Path]]):
    ds: Dataset = Dataset()
    for d_spec in ds_spec:
        ds += _import_dataset(d_spec)

    ds_df = ds.to_df()

    with create_file(
        loading_filepath(name=name, filename=IMPORTED_DATASET_FILENAME), mode="wb", encoding=None, buffering=-1
    ) as f:
        f.write(pickle.dumps(ds_df))


def load_dataset(name: str):
    with open(loading_filepath(name=name, filename=IMPORTED_DATASET_FILENAME), mode="rb") as pickle_file:
        dataset = pickle.load(pickle_file)
    return dataset
