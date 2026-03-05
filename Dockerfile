FROM python:3.13-slim

WORKDIR /app

# System deps for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install --no-cache-dir .

COPY examples/ examples/

EXPOSE 8501 8000
