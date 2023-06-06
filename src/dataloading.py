from typing import List, Tuple

from model import (
    ImportDataSet,
    Token,
    Code,
    Sentence,
    Doc,
    Dataset,
)


def _create_instances(
    imported_dataset: ImportDataSet,
) -> Tuple[List[Doc], List[Token], List[Code]]:
    codes = [
        Code.construct(index=c.index, name=c.name, tore_index=c.tore)
        for c in imported_dataset.codes
    ]

    tokens = [
        Token.construct(index=t.index, name=t.name, lemma=t.lemma, pos=t.pos)
        for t in imported_dataset.tokens
    ]

    docs = [Doc.construct(name=d.name) for d in imported_dataset.docs]

    return docs, tokens, codes


def _create_connections(
    imported_dataset: ImportDataSet,
    docs: List[Doc],
    tokens: List[Token],
    codes: List[Code],
) -> Tuple[List[Doc], List[Token], List[Code]]:
    # doc -> token
    for imported_doc, doc in zip(imported_dataset.docs, docs):
        doc.content = [
            tokens[i]
            for i in range(imported_doc.begin_index, imported_doc.end_index)
        ]

    for imported_code, code in zip(imported_dataset.codes, codes):
        for i in imported_code.tokens:
            # token -> code
            tokens[i].tore_codes.append(code)
            # code -> token
            code.tokens.append(tokens[i])

    return docs, tokens, codes


def _split_document_into_sentences(
    docs: List[Doc],
) -> Tuple[List[Doc], List[Sentence]]:
    # split content into sentences
    sentences = []
    punctuation = [".", "!", "?"]
    for doc in docs:
        starts: List[int] = [0]
        ends: List[int] = [len(doc.content)]
        shift_reg: List[str] = [" ", " ", " "]

        for idx, token in enumerate(doc.content):
            # Handle sentence terminator
            shift_reg.insert(0, token.name)
            shift_reg.pop()

            if "".join(shift_reg) == "###":
                ends.append(idx - 2)
                starts.append(idx + 1)

            # Handle punctuation
            if token.name in punctuation:
                try:
                    if doc.content[idx + 1].name not in punctuation:
                        starts.append(idx + 1)
                        ends.append(idx + 1)
                except IndexError:
                    pass

        starts.sort()
        ends.sort()

        docs_sentences: List[Sentence] = []
        for start, end in zip(starts, ends):
            if end - start != 0:
                s = Sentence(tokens=[t for t in doc.content[start:end]])
                docs_sentences.append(s)

        doc.sentences = docs_sentences
        sentences += docs_sentences

    return docs, sentences


def denormalize_dataset(
    imported_dataset: ImportDataSet,
) -> Dataset:
    # create instances
    docs, tokens, codes = _create_instances(imported_dataset=imported_dataset)

    # recreate relationship
    docs, tokens, codes = _create_connections(
        imported_dataset=imported_dataset,
        docs=docs,
        tokens=tokens,
        codes=codes,
    )

    docs, sentences = _split_document_into_sentences(docs=docs)

    return Dataset(
        docs=docs,
        tokens=tokens,
        codes=codes,
        sentences=sentences,
    )
