FROM python:3.11-slim-buster

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    nginx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
WORKDIR /app
RUN pip3 install --no-cache-dir --upgrade pip -r requirements.txt

RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN python -m nltk.downloader wordnet
RUN python -m nltk.downloader omw-1.4

COPY . /app/
COPY src/service/nginx.conf /etc/nginx
RUN pip3 install --no-cache-dir --upgrade pip -r requirements-local.txt

WORKDIR /app/src/service/
RUN chmod +x start.sh
EXPOSE 80

CMD ["./start.sh"]
