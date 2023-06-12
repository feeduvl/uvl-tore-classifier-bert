from typing import List
from pydantic import ValidationError
from .model import (
    ImportDataSet,
    Token,
    Code,
    Sentence,
    Dataset,
)


def split_tokenlist_into_sentences(
    tokens: List[Token],
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
            s = Sentence(tokens=[t for t in tokens[start:end]])
            doc_sentences.append(s)

    return doc_sentences


def denormalize_dataset(
    imported_dataset: ImportDataSet,
) -> Dataset:
    tokenindex_codes: dict[int, List[Code]] = {}
    for imported_code in imported_dataset.codes:
        if imported_code.index is not None:
            try:
                code = Code(
                    index=imported_code.index,
                    name=imported_code.name,
                    tore_index=imported_code.tore,
                )
                for token_id in imported_code.tokens:
                    try:
                        tokenindex_codes[token_id]
                    except KeyError:
                        tokenindex_codes[token_id] = []

                    tokenindex_codes[token_id].append(code)
            except ValidationError:
                pass

    sentences: List[Sentence] = []
    for imported_document in imported_dataset.docs:
        tokens: List[Token] = []
        for token_index in range(
            imported_document.begin_index, imported_document.end_index
        ):
            imported_token = imported_dataset.tokens[token_index]
            if imported_token.index is not None:
                try:
                    token = Token(
                        index=imported_token.index,
                        name=imported_token.name,
                        lemma=imported_token.lemma,
                        pos=imported_token.pos,
                        tore_codes=tokenindex_codes[imported_token.index],
                    )
                    tokens.append(token)
                except KeyError as e:
                    if imported_token.num_tore_codes == 0:
                        pass
                    else:
                        raise e

        new_sentences = split_tokenlist_into_sentences(tokens=tokens)
        sentences += new_sentences

    return Dataset(sentences=sentences)
