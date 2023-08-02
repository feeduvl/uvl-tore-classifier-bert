FROM python:3.11-slim-buster

ENV PYDEVD_DISABLE_FILE_VALIDATION=1
ENV GLIBC_TUNABLES=glibc.rtld.optional_static_tls=1024
ENV CUDA_VISIBLE_DEVICES=-1

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    nginx \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

COPY requirements.txt /app/

RUN pip3 install --no-cache-dir  -r /app/requirements.txt

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4

WORKDIR /app
COPY . /app/
COPY src/service/nginx.conf /etc/nginx
RUN pip3 install --no-cache-dir --upgrade pip -r requirements-local.txt

ARG mlflow_tracking_password
ARG mlflow_tracking_username
ARG mlflow_tracking_uri

ENV MLFLOW_TRACKING_USERNAME=$mlflow_tracking_username
ENV MLFLOW_TRACKING_PASSWORD=$mlflow_tracking_password
ENV MLFLOW_TRACKING_URI=$mlflow_tracking_uri

ENV UVL_BERT_RUN_EXPERIMENTS=False
ENV UVL_BERT_PIN_COMMITS=False
ENV MPLCONFIGDIR=/app/temp/matplotlib/
ENV TRANSFORMERS_CACHE=/app/temp/transformers/

RUN mkdir -p $MPLCONFIGDIR
RUN mkdir -p $TRANSFORMERS_CACHE

RUN jupyter nbconvert --to python --execute train.ipynb

WORKDIR /app/src/service/
RUN chmod +x start.sh

EXPOSE 9693

CMD ["./start.sh"]
