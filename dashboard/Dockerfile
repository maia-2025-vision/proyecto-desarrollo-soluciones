# -------------------------
# Etapa 1: builder
# -------------------------

FROM python:3.13-slim AS builder

ENV PYTHONBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

RUN adduser --disabled-password --gecos '' ui-user

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Install dependencies
COPY .python-version ./
COPY pyproject.toml ./
COPY README.md ./
COPY uv.lock ./
COPY api/ api/
COPY dashboard/ dashboard/


# Install dependencies only (no dev dependencies needed for streamlit)
RUN --mount=type=cache,id=s/fe91e38d-e9bd-4b56-a9de-4861ff90ad4d-/tmp/uv-cache,target=/tmp/uv-cache \
    uv sync --frozen --link-mode=copy --no-dev --no-install-project --only-group dashboard-deps

RUN chown -R ui-user:ui-user /app


# -------------------------
# Etapa 2: final
# -------------------------
FROM python:3.13-slim

ENV PYTHONBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN adduser --disabled-password --gecos '' ui-user

WORKDIR /app

COPY --from=builder /app /app

RUN chown -R ui-user:ui-user /app

USER ui-user

# Expose Streamlit default port
EXPOSE 8501

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Run Streamlit app
CMD ["/app/.venv/bin/streamlit", "run", "dashboard/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
