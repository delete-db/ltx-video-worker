FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git curl ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# PyTorch
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install \
    --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

# ltx-pipelines (direct from GitHub)
RUN git clone https://github.com/Lightricks/LTX-2.git /tmp/LTX-2 \
    && python -m pip install -e /tmp/LTX-2/packages/ltx-core \
    && python -m pip install -e /tmp/LTX-2/packages/ltx-pipelines

# RunPod + utilities (pin transformers — 5.x breaks Gemma3TextConfig)
RUN python -m pip install runpod requests Pillow transformers==4.52.0

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
