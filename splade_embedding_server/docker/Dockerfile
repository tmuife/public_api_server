FROM python:3.12

WORKDIR /code

COPY ./pyproject.toml /code/pyproject.toml
COPY ./uv.lock /code/uv.lock
COPY ./.env /code/.env
COPY ./main.py /code/main.py
COPY ./app /code/app

COPY ./certs/localhost.crt /etc/ssl/certs/
COPY ./certs/localhost.key /etc/ssl/certs/

COPY ./models /code/models

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv
RUN uv sync --frozen --no-dev --no-install-project

HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/docs || exit 1

CMD [".venv/bin/python", "main.py"]
