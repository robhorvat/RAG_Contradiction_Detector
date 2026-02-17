FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Keep app dependencies deterministic and choose torch wheel source explicitly.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN grep -vE '^torch([<=>].*)?$' /app/requirements.txt > /tmp/requirements.no_torch.txt \
    && python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.no_torch.txt \
    && python -m pip install --index-url ${TORCH_INDEX_URL} torch

COPY . /app

EXPOSE 8501 9108

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
