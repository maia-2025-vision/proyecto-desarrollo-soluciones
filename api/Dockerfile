FROM python:3.13-slim

ENV PYTHONBUFFERED=1 \
    PYTHONDONTWRITEBYCODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

# Crear usuario que ejecuta la app
# RUN adduser --disabled-password --gecos '' api-user

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Definir directorio de trabajo
WORKDIR /opt/cow-detect

# Instalar dependencias
COPY .python-version .
COPY README.md .
COPY uv.lock .
COPY pyproject.toml .

# Model weights copied into image here:`
# If changing model path change the following TWO LINES!
COPY data/training/teo/v1/faster-rcnn/model.pth  data/training/teo/v1/faster-rcnn/model.pth
ENV MODEL_PATH=data/training/teo/v1/faster-rcnn/model.pth
###
RUN find ./data

RUN pwd
RUN ls -lrt

RUN uv cache dir
# Run with mounted cached so that we don´t download all packages everytime...
# Install dependencies only but not the project
RUN --mount=type=cache,target=/tmp/uv-cache \
   uv sync --frozen --link-mode=copy --no-dev --no-install-project --group torch-train

COPY cow_detect/  cow_detect/
COPY api/ api/

RUN ls -lrt

# Hacer el directorio de trabajo ejecutable
# RUN chown -R api-user:api-user ./

USER root

EXPOSE 8000

ENV UVICORN_PORT=8000
ENV UVICORN_RELOAD=false


RUN uv pip install -e "."

CMD ["uv", "run", "--no-sync", "python", "api/run_api.py"]
