from typing import List
from typing import Literal
from typing import Union

import pandas as pd
from nltk import download
from nltk import pos_tag
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from transformers import BertTokenizerFast

from classifiers.bert.classifier import create_bert_dataset
from classifiers.bert.classifier import create_staged_bert_dataset
from classifiers.bilstm.classifier import classify_with_bilstm
from classifiers.sner.classifier import classify_with_sner
from classifiers.staged_bert.classifier import classify_with_bert
from classifiers.staged_bert.classifier import classify_with_bert_stage_1
from classifiers.staged_bert.classifier import classify_with_bert_stage_2
from service.service_types import Classifier_Options
from service.service_types import Code
from service.service_types import Documents
from service.service_types import Label2Id2Label
from service.service_types import Models
from tooling.model import Label_None_Pad, ZERO
from tooling.model import string_list_lists_to_datadf


def do_nltk_downloads() -> None:
    download("punkt")
    download("averaged_perceptron_tagger")
    download("wordnet")
    download("omw-1.4")


def get_wordnet_pos(
    treebank_tag: str,
) -> Union[
    nltk.corpus.wordnet.ADJ,
    nltk.corpus.wordnet.VERB,
    nltk.corpus.wordnet.NOUN,
    nltk.corpus.wordnet.ADV,
    Literal[""],
]:
    if treebank_tag.startswith("J"):
        return nltk.corpus.wordnet.ADJ
    if treebank_tag.startswith("V"):
        return nltk.corpus.wordnet.VERB
    if treebank_tag.startswith("N"):
        return nltk.corpus.wordnet.NOUN
    if treebank_tag.startswith("R"):
        return nltk.corpus.wordnet.ADV
    return ""


def get_tokens(documents: Documents) -> List[List[str]]:
    do_nltk_downloads()  # should be already run in the final container

    tokenized_sent = [
        item for doc in documents for item in sent_tokenize(doc["text"])
    ]
    tokenized_docs = [word_tokenize(sent) for sent in tokenized_sent]

    return tokenized_docs


def get_lemmas(tokens: List[List[str]]) -> List[List[str]]:
    do_nltk_downloads()
    all_lemmas = []  # should be already run in the final container
    lemmatizer = WordNetLemmatizer()

    for sentence in tokens:
        pos_tags = [get_wordnet_pos(tup[1]) for tup in pos_tag(sentence)]
        lemmas = [
            lemmatizer.lemmatize(token, pos=pos_tags[ind]).lower()
            if pos_tags[ind] != ""
            else token.lower()
            for ind, token in enumerate(sentence)
        ]
        all_lemmas.append(lemmas)

    return all_lemmas


def build_codes(
    lemmas: List[List[str]],
    labels: List[List[Label_None_Pad]],
) -> List[Code]:
    token_idx = 0
    code_idx = 0
    codes = []
    for sentence_lemmas, sentence_labels in zip(lemmas, labels, strict=True):
        for lemma, label in zip(sentence_lemmas, sentence_labels):
            if label is ZERO:
                token_idx += 1
                continue

            label = label.replace("_", " ")

            code = Code(
                tokens=[token_idx],
                name=lemma,
                tore=label,
                index=code_idx,
                relationship_memberships=[],
            )
            codes.append(code)
            code_idx += 1
            token_idx += 1
    return codes


def classify_dataset(
    documents: Documents,
    models: Models,
    method: Classifier_Options,
    label2id2label: Label2Id2Label,
    max_len: int,
    glove_model: pd.DataFrame,
) -> List[Code]:
    tokens = get_tokens(documents=documents)
    lemmas = get_lemmas(tokens=tokens)

    data = string_list_lists_to_datadf(tokens)

    if method == "bert-classifier/sner_bert":
        tokenizer = BertTokenizerFast.from_pretrained(models.bert_2_sner)

        hinted_data = classify_with_sner(model_path=models.sner, data=data)

        hinted_bert_data = create_staged_bert_dataset(
            input_data=hinted_data,
            label2id=label2id2label.label2id,
            hint_label2id=label2id2label.hint_label2id,
            tokenizer=tokenizer,
            max_len=max_len,
            ignore_labels=True,
            align_labels=False,
        )

        labels = classify_with_bert_stage_2(
            model_path=models.bert_2_sner,
            bert_data=hinted_bert_data,
            id2label=label2id2label.id2label,
        )

    elif method == "bert-classifier/bilstm_bert":
        tokenizer = BertTokenizerFast.from_pretrained(models.bert_2_bilstm)

        hinted_data = classify_with_bilstm(
            model_path=models.bilstm,
            data=data,
            max_len=max_len,
            glove_model=glove_model,
            hint_id2label=label2id2label.hint_id2label,
        )

        hinted_bert_data = create_staged_bert_dataset(
            input_data=hinted_data,
            label2id=label2id2label.label2id,
            hint_label2id=label2id2label.hint_label2id,
            tokenizer=tokenizer,
            max_len=max_len,
            ignore_labels=True,
            align_labels=False,
        )

        labels = classify_with_bert_stage_2(
            model_path=models.bert_2_bilstm,
            bert_data=hinted_bert_data,
            id2label=label2id2label.id2label,
        )

    elif method == "bert-classifier/bert_bert":
        tokenizer_1 = BertTokenizerFast.from_pretrained(models.bert_1)

        hinted_bert_data = classify_with_bert_stage_1(
            model_path=models.bert_1,
            data=data,
            label2id=label2id2label.label2id,
            tokenizer=tokenizer_1,
            max_len=max_len,
        )

        labels = classify_with_bert_stage_2(
            model_path=models.bert_2_bert,
            bert_data=hinted_bert_data,
            id2label=label2id2label.id2label,
        )

    elif method == "bert-classifier/bert":
        tokenizer = BertTokenizerFast.from_pretrained(models.bert)

        bert_data = create_bert_dataset(
            input_data=data,
            label2id=label2id2label.label2id,
            tokenizer=tokenizer,
            max_len=max_len,
            ignore_labels=True,
            align_labels=False,
        )

        labels = classify_with_bert(
            model_path=models.bert,
            bert_data=bert_data,
            id2label=label2id2label.id2label,
        )
    else:
        raise NotImplementedError

    return build_codes(lemmas=lemmas, labels=labels)
