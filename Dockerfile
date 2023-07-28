FROM python:3.11-slim-buster

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    nginx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
WORKDIR /app
RUN pip3 install --no-cache-dir --upgrade pip -r requirements.txt

COPY . /app/
COPY src/service/nginx.conf /etc/nginx
RUN pip3 install --no-cache-dir --upgrade pip -r requirements-local.txt

WORKDIR /app/src/service/
RUN chmod +x start.sh
EXPOSE 80

CMD ["./start.sh"]
