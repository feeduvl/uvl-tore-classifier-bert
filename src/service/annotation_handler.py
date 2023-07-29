from logging import Logger
from typing import cast
from typing import List

import requests

from service.types import Annotation
from service.types import Code


def initialize_annotation(
    dataset_name: str, annotation_name: str, logger: Logger
) -> int:
    logger.info(
        f"Initialize annotation {annotation_name} of dataset {dataset_name}"
    )

    annotation = {"name": annotation_name, "dataset": dataset_name}
    request = requests.post(
        "https://feed-uvl.ifi.uni-heidelberg.de/hitec/orchestration/concepts/annotationinit/",
        json=annotation,
    )

    return request.status_code


def get_annotation(annotation_name: str, logger: Logger) -> Annotation:
    logger.info("Get created annotation")
    request = requests.get(
        f"https://feed-uvl.ifi.uni-heidelberg.de/hitec/repository/concepts/annotation/name/{annotation_name}"
    )
    return cast(Annotation, request.json())


def add_codes_to_tokens(
    annotation: Annotation, codes: List[Code]
) -> Annotation:
    for code in codes:
        index = code["tokens"][0]
        annotation["tokens"][index]["num_name_codes"] = 1
        annotation["tokens"][index]["num_tore_codes"] = 1
    return annotation


def add_classification_to_annotation(
    annotation_name: str, codes: List[Code], logger: Logger
) -> Annotation:
    annotation = get_annotation(annotation_name, logger)
    annotation["codes"] = codes
    return add_codes_to_tokens(annotation, codes)


def store_annotation(annotation: Annotation, logger: Logger) -> int:
    logger.info("Storing annotation")

    request = requests.post(
        "https://feed-uvl.ifi.uni-heidelberg.de/hitec/repository/concepts/store/annotation/",
        json=annotation,
    )

    return request.status_code


def create_new_annotation(
    dataset_name: str, annotation_name: str, codes: List[Code], logger: Logger
) -> None:
    status_code = initialize_annotation(dataset_name, annotation_name, logger)
    if status_code == 200:
        annotation = add_classification_to_annotation(
            annotation_name, codes, logger
        )
        store_annotation(annotation, logger)
