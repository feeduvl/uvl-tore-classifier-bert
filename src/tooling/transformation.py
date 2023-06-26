import pandas as pd
from tooling.model import Sentence
from tooling.model import Token


def transform_token(token: Token) -> Token:
    codes = token.tore_codes
    if codes:
        new_codes = codes
        new_token = Token(
            index=token.index,
            name=token.name,
            lemma=token.lemma,
            pos=token.pos,
            tore_codes=new_codes,
        )

        return new_token
    return token


def transform_row(row: pd.DataFrame) -> pd.DataFrame:
    sentence: Sentence = row["self"]
    new_token_list = list(map(transform_token, sentence.tokens))

    sentence = Sentence(tokens=new_token_list, source=sentence.source)
    row["self"] = sentence

    return row


def transform_dataset(ds: pd.DataFrame) -> pd.DataFrame:
    return ds.apply(transform_row, axis=1)
