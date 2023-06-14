import pytest
from typing import List
from data import Token, split_tokenlist_into_sentences
from pathlib import Path
from data import ImportDataSet, denormalize_dataset


def generate_doc(content: str) -> List[Token]:
    tokens = []
    for idx, letter in enumerate(content):
        tokens.append(Token(index=idx, name=letter, lemma=letter, pos=""))
    return tokens


@pytest.mark.parametrize(
    "content,sentence_length",
    [
        ("ABC###", [3]),  # Single Sentence without punctiation
        ("ABC.###", [4]),  # Single Sentence with a dot
        ("ABC?###", [4]),  # Single Sentence with a question mark
        ("ABC!###", [4]),  # Single Sentence with an exklamation mark
        ("ABC###123###", [3, 3]),  # Doube Sentence without punctuation
        ("ABC.###123.###", [4, 4]),  # Doube Sentence with a dot
        ("ABC?###123?###", [4, 4]),  # Doube Sentence with a question mark
        ("ABC!###123!###", [4, 4]),  # Doube Sentence with an exklamation mark
        ("ABC#123!###", [8]),  # Sentence with a hashtag
        ("ABC##123!###", [9]),  # Sentence with a double hashtag
        ("ABC...###", [6]),  # Single Sentence with a dot
        ("ABC...123###", [6, 3]),  # Single Sentence with a dot
        ("ABC...123.###", [6, 4]),  # Single Sentence with a dot
        ("ABC.", [4]),  # Single Sentence with a dot and no sentence marker
        (
            "ABC!",
            [4],
        ),  # Single Sentence with an exclamation mark and no sentence marker
        (
            "ABC?",
            [4],
        ),  # Single Sentence with a questionmark and no sentence marker
        (
            "ABC.ABC.",
            [4, 4],
        ),  # Two sentences with a dot and no sentence marker
        (
            "ABC!ABC!",
            [4, 4],
        ),  # Single Sentence with an exclamation mark and no sentence marker
        (
            "ABC?ABC?",
            [4, 4],
        ),  # Single Sentence with a questionmark and no sentence marker
        ("ABC...", [6]),  # Single Sentence with an elipsis
        ("ABC...123", [6, 3]),  # Single Sentence with an elipsis in the middle
        (
            "ABC...123.",
            [6, 4],
        ),  # Single Sentence with an elipsis in the middle and a dot at the end
    ],
)
def test_eval(content: str, sentence_length: List[int]):
    tokens = generate_doc(content)
    sentences = split_tokenlist_into_sentences(tokens)

    print(sentences)

    assert len(sentences) == len(sentence_length)

    for res, exp in zip(sentences, sentence_length):
        assert len(res.tokens) == exp


def test_dataloading():
    BASE_PATH = Path(__file__).parent
    FILE_PATH = BASE_PATH.joinpath(Path("./test_dataset.json"))

    imported_ds = ImportDataSet.parse_file(FILE_PATH.resolve())
    denormalized_ds = denormalize_dataset(imported_dataset=imported_ds)
    assert (
        str(denormalized_ds.sentences[0])
        == "Is there any extension I can add to chrome which gives a warning before closing all tabs in the window by mistake ?"
    )
    assert (
        str(denormalized_ds.sentences[1])
        == "For example : I have multiple tabs open , by mistake I click on windows cross to close the window , is there something I can add to chrome that it might give a warning like , If I want to close the all tabs in window or only the open tab , instead of directly closing all the tabs open in the window"
    )
