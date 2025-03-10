FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /transformer-from-scratch
COPY pyproject.toml .
COPY exercises/__init__.py exercises/__init__.py

RUN uv sync --extra gpu
RUN uv sync --extra gpu --extra flash
