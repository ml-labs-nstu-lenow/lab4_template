FROM python:3.12-slim-bookworm AS builder

RUN mkdir app
WORKDIR /app

COPY poetry.lock pyproject.toml ./
RUN pip install --no-cache-dir poetry \
  && poetry config virtualenvs.in-project true \
  && poetry install --with dev --no-interaction --no-ansi

FROM python:3.12-slim-bookworm

COPY --from=builder /app /app

WORKDIR /app

# set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:${PATH}"
