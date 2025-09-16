FROM python:3.13-slim

ENV PYTHONBUFFERED=1 \
    PYTHONDONTWRITEBYCODE=1 \
    UV_CACHE_DIR=/tmp/uv-cache

# Copy uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /opt/cow-detect

# Install dependencies
COPY .python-version .
COPY README.md .
COPY uv.lock .
COPY pyproject.toml .

# Install dependencies only (no dev dependencies needed for streamlit)
RUN --mount=type=cache,target=/tmp/uv-cache \
   uv sync --frozen --link-mode=copy --no-dev --no-install-project

# Copy application code
COPY streamlit_app.py .
COPY pages/ pages/
COPY api/ api/

# Install the package
RUN uv pip install -e "."

# Expose Streamlit default port
EXPOSE 8501

# Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Run Streamlit app
CMD ["uv", "run", "--no-sync", "streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]