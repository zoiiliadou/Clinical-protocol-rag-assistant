FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt .

RUN pip install --no-cache-dir --no-compile --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.docker.txt \
    && find /usr/local/lib/python3.12 -name "*.pyc" -delete \
    && find /usr/local/lib/python3.12 -type d -name "__pycache__" -exec rm -rf {} +

COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000
EXPOSE 8501
