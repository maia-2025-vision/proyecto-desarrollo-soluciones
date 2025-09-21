# -------------------------
# Etapa 1: builder
# -------------------------
FROM python:3.13-slim AS builder

ENV PYTHONBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

RUN apt-get update && apt-get install -y git && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN adduser --disabled-password --gecos '' api-user

RUN pip install uv


WORKDIR /app

COPY .python-version ./
COPY pyproject.toml ./
COPY README.md ./
COPY uv.lock ./
COPY dvc.yaml ./
COPY dvc.lock ./
COPY .dvc/ .dvc/
COPY .dvcignore ./
COPY data/.gitignore data/.gitignore
COPY api/ api/

RUN uv cache dir

RUN --mount=type=cache,id=s/4ea0a428-f8cf-4f69-a0dd-68c65a55469b-/tmp/uv-cache,target=/tmp/uv-cache \
    uv sync --frozen --link-mode=copy --no-dev --no-install-project --only-group api-deps


RUN git init && \
    uv run --no-sync dvc remote modify aws-remote --unset profile && \
    uv run --no-sync dvc pull data/training/teo/v1/faster-rcnn/model.pth && \
    rm -rf .dvc .git


RUN uv pip install torch torchvision --native-tls --index-url https://download.pytorch.org/whl/cpu
    

ENV MODEL_PATH=data/training/teo/v1/faster-rcnn/model.pth

RUN chown -R api-user:api-user /app


# -------------------------
# Etapa 2: final
# -------------------------
FROM python:3.13-slim

ENV MODEL_PATH=data/training/teo/v1/faster-rcnn/model.pth \
    UVICORN_PORT=8000 \
    UVICORN_RELOAD=false \
    PYTHONBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN adduser --disabled-password --gecos '' api-user

WORKDIR /app

COPY --from=builder /app /app

RUN chown -R api-user:api-user /app

USER api-user

EXPOSE 8000

CMD ["/app/.venv/bin/python", "api/run_api.py"]