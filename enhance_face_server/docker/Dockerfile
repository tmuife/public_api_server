FROM cnstark/pytorch:2.0.1-py3.10.11-cuda11.8.0-ubuntu22.04

WORKDIR /code

COPY ./poetry.lock /code/poetry.lock
COPY ./pyproject.toml /code/pyproject.toml
COPY ./poetry.toml /code/poetry.toml
COPY ./.env /code/.env
COPY ./main.py /code/main.py
COPY ./app /code/app

COPY ./certs/localhost.crt /etc/ssl/certs/
COPY ./certs/localhost.key /etc/ssl/certs/

COPY ./models /code/models
COPY ./gfpgan /code/gfpgan

RUN apt-get update && apt-get install -y && apt-get install ffmpeg -y \
    build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install poetry
#RUN poetry export -f requirements.txt --output requirements.txt --without-hashes
#RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN poetry install --no-cache

HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/docs || exit 1

CMD ["poetry", "run", "python", "main.py"]
