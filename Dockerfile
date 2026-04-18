FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ARG GPT_SOVITS_REPO=https://github.com/RVC-Boss/GPT-SoVITS.git
ARG GPT_SOVITS_REF=2d9193b0d3c0eae0c3a14d8c68a839f1bae157dc

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GPT_SOVITS_ROOT=/opt/GPT-SoVITS \
    MODEL_VERSION=v2ProPlus \
    LOAD_MODEL_AT_STARTUP=true \
    DEVICE=cuda \
    IS_HALF=true \
    OUTPUT_MODE=base64 \
    DEFAULT_MEDIA_TYPE=wav \
    MODEL_CACHE_DIR=/runpod-volume/gpt-sovits

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ffmpeg \
        git \
        libsndfile1 \
        libsox-dev \
        sox \
    && rm -rf /var/lib/apt/lists/*

RUN git clone "${GPT_SOVITS_REPO}" "${GPT_SOVITS_ROOT}" \
    && cd "${GPT_SOVITS_ROOT}" \
    && git checkout "${GPT_SOVITS_REF}"

COPY requirements-gpt-sovits-inference.txt requirements-serverless.txt pyproject.toml /app/
COPY src /app/src
COPY handler.py /app/handler.py

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/requirements-gpt-sovits-inference.txt \
    && python -m pip install -r /app/requirements-serverless.txt \
    && python -m pip install -e /app

CMD ["python", "-u", "/app/handler.py"]
