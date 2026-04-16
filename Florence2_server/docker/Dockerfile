FROM python:3.12

WORKDIR /code

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY ./uv.lock /code/uv.lock
COPY ./pyproject.toml /code/pyproject.toml
COPY ./.env /code/.env
COPY ./main.py /code/main.py
COPY ./app /code/app

COPY ./certs/localhost.crt /etc/ssl/certs/
COPY ./certs/localhost.key /etc/ssl/certs/

RUN apt-get update && apt-get install -y ffmpeg \
    build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN uv sync --frozen --no-dev

HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:8000/docs || exit 1

CMD ["uv", "run", "python", "main.py"]
