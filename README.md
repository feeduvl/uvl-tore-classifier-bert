# Setup

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-local.txt
```

Optionally

```sh
pip install -r requirements-mac.txt
```

You need to supply the following environment variables for the application to work.
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
MLFLOW_TRACKING_URI=

I suggest using a tool like dotenv for this.

# Folder Structure

## src

Contains the reusable application pieces:

- tooling to access the datasets
- the classifier implementations
- the experiment pipelines
- the service used in the feed.uvl project
- tests
- helpers

Experiments are started by running the shell scripts directly

## Evaluation

Evaluation contains a set of jupyter notebooks that where used to create the evaluation.

## Training

The models for the service are trained via the `train.ipynb` notebook in the project root.

The service container is build by running:
`docker build -t $CONTAINER_NAME -f "./Dockerfile" --build-arg mlflow_tracking_username=XXXXXX --build-arg mlflow_tracking_password=XXXXXX --build-arg mlflow_tracking_uri=XXXXXX  .  `
replacing XXXXXX with the appropriate credentials.