from typing import List, Tuple, TypedDict

from model import (
    ImportDataSet,
    Token,
    Code,
    Sentence,
    Doc,
    Dataset,
)


def _split_document_into_sentences(
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


def denormalize_dataset_v2(
    imported_dataset: ImportDataSet,
) -> Dataset:
    tokenindex_codes: dict[int, List[Code]] = {}
    for imported_code in imported_dataset.codes:
        code = Code(
            index=imported_code.index,
            name=imported_code.name,
            tore_index=imported_code.tore,
        )
        for token_id in imported_code.tokens:
            if tokenindex_codes[token_id] is None:
                tokenindex_codes[token_id] = []

            tokenindex_codes[token_id].append(code)

    docs = []
    for imported_document in imported_dataset.docs:
        tokens: List[Token] = []
        for token_index in range(
            imported_document.begin_index, imported_document.end_index
        ):
            imported_token = imported_dataset.tokens[token_index]
            token = Token(
                index=imported_token.index,
                name=imported_token.name,
                lemma=imported_token.lemma,
                pos=imported_token.pos,
                tore_codes=tokenindex_codes[imported_token.index],
            )
            tokens.append(token)

        sentences = _split_document_into_sentences(tokens=tokens)

        doc = Doc(name=imported_document.name, sentences=sentences)
        docs.append(doc)

    return Dataset(docs=docs)
