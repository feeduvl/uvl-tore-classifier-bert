from logging.config import dictConfig
from typing import cast
from typing import get_args

from flask import Flask
from flask import json
from flask import jsonify
from flask import request
from flask import Response

from service.annotation_handler import create_new_annotation
from service.classifier import classify_dataset
from service.config import configure
from service.service_types import Classifier_Options
from service.service_types import Documents
from typing import Dict, Any


app = Flask(__name__)

cache: Dict[str, Any] = {}
cache = configure(cache)


@app.route(
    "/hitec/classify/concepts/bert-classifier/bert/run", methods=["POST"]
)
@app.route(
    "/hitec/classify/concepts/bert-classifier/bert_bert/run", methods=["POST"]
)
@app.route(
    "/hitec/classify/concepts/bert-classifier/bilstm_bert/run",
    methods=["POST"],
)
@app.route(
    "/hitec/classify/concepts/bert-classifier/sner_bert/run", methods=["POST"]
)
def classify_tore() -> Response:
    app.logger.info("BERT Classification run requested")
    app.logger.debug("/hitec/classify/concepts/bert-classifier/run called")

    content = json.loads(request.data.decode("utf-8"))
    documents = cast(Documents, content["dataset"]["documents"])

    # access the "per worker" cache of configuration
    global cache
    cache = configure(cache)

    app.logger.info(content)

    method = content["method"]

    if method not in get_args(Classifier_Options):
        raise ValueError(
            f"{method} is not a valid option from {get_args(Classifier_Options)}"
        )
    app.logger.info(f"{method=} selected")

    app.logger.debug("First Sentence")
    app.logger.debug(documents[0])

    codes = classify_dataset(
        documents=documents,
        models=cache["models"],
        label2id2label=cache["label2id2label"],
        method=method,
        max_len=cache["max_len"],
        glove_model=cache["glove_model"],
    )

    if content["params"]["persist"] == "true":
        app.logger.info(f"Create annotations settings: {True}")
        dataset_name: str = content["dataset"]["name"]
        annotation_name: str = content["params"]["annotation_name"]

        create_new_annotation(dataset_name, annotation_name, codes, app.logger)
    else:
        app.logger.info(f"Create annotations settings: {False}")

    result = dict()
    result.update({"codes": codes})

    app.logger.info(result)

    return jsonify(result)


@app.route("/hitec/classify/concepts/bert-classifier/status", methods=["GET"])
def get_status() -> Response:
    try:
        # access the "per worker" cache of configuration
        global cache
        cache = configure(cache)

        status = {
            "status": "operational",
        }
    except Exception as e:
        status = {"status": "not_operational", "error": str(e)}
    return jsonify(status)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
