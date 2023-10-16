FROM python:3.11-slim-buster

ENV PYDEVD_DISABLE_FILE_VALIDATION=1
ENV GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2048
ENV CUDA_VISIBLE_DEVICES=-1

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    nginx \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

COPY requirements/requirements.txt /app/

RUN pip3 install --no-cache-dir  -r /app/requirements.txt

RUN mkdir -p /usr/share/nltk_data
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt
RUN python -m nltk.downloader -d /usr/share/nltk_data averaged_perceptron_tagger
RUN python -m nltk.downloader -d /usr/share/nltk_data wordnet
RUN python -m nltk.downloader -d /usr/share/nltk_data omw-1.4



WORKDIR /app
COPY . /app/
COPY src/service/nginx.conf /etc/nginx
RUN pip3 install --no-cache-dir --upgrade pip -r requirements/requirements-local.txt

ARG mlflow_tracking_password
ARG mlflow_tracking_username
ARG mlflow_tracking_uri

ENV MLFLOW_TRACKING_USERNAME=$mlflow_tracking_username
ENV MLFLOW_TRACKING_PASSWORD=$mlflow_tracking_password
ENV MLFLOW_TRACKING_URI=$mlflow_tracking_uri

ENV UVL_BERT_RUN_EXPERIMENTS=False
ENV UVL_BERT_PIN_COMMITS=False

ENV MPLCONFIGDIR=/app/temp/matplotlib/
RUN mkdir -p $MPLCONFIGDIR

ENV TRANSFORMERS_CACHE=/app/temp/transformers/
RUN mkdir -p $TRANSFORMERS_CACHE

RUN jupyter nbconvert --to python --execute train.ipynb

WORKDIR /app/src/service/

RUN export LD_PRELOAD=/usr/local/lib/python3.11/site-packages/tensorflow_cpu_aws.libs/libgomp-cc9055c7.so.1.0.0

RUN chmod +x start.sh

ENV HF_HOME=/usr/share/huggingface
RUN mkdir -p /usr/share/huggingface

ENV GENSIM_DATA_DIR=/usr/share/gensim-data
RUN mkdir -p /usr/share/gensim-data

EXPOSE 9694

CMD ["./start.sh"]
