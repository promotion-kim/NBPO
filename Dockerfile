FROM docker.io/vllm/vllm-openai:v0.5.1

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/RONPO

RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/RONPO
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/bin/bash"]
