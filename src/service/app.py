from logging.config import dictConfig

from flask import Flask
from flask import json
from flask import jsonify
from flask import request
from flask import Response

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "wsgi": {
                "class": "logging.StreamHandler",
                "stream": "sys.stdout",
                "formatter": "default",
            }
        },
        "root": {"level": "DEBUG", "handlers": ["wsgi"]},
    }
)


app = Flask(__name__)


@app.route("/hitec/classify/concepts/bert-classifier/run", methods=["POST"])
def classify_tore() -> Response:
    app.logger.info("BERT Classification run requested")
    app.logger.debug("/hitec/classify/concepts/bert-classifier/run called")

    content = json.loads(request.data.decode("utf-8"))

    documents = content["dataset"]["documents"]

    app.logger.info(documents)

    # codes = classifyDataset(documents)

    if content["params"]["persist"] == "true":
        app.logger.info(f"Create annotations settings: {True}")
        dataset_name: str = content["dataset"]["name"]
        annotation_name: str = content["params"]["annotation_name"]

        # createNewAnnotation(dataset_name, annotation_name, codes, app.logger)
    else:
        app.logger.info(f"Create annotations settings: {False}")

    result = dict()
    # result.update({"codes": codes})
    return jsonify(result)


@app.route("/hitec/classify/concepts/bert-classifier/status", methods=["GET"])
def get_status() -> Response:
    status = {
        "status": "operational",
    }

    return jsonify(status)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9693)
