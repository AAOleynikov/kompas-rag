FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

COPY configs ./configs
COPY src ./src

EXPOSE 7860

CMD ["python", "-m", "rag.web_ui", "--config", "configs/rag.yaml", "--persist-dir", "data/indexes/chroma_db", "--host", "0.0.0.0", "--port", "7860"]
