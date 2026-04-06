FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV COMFY_ROOT=/opt/ComfyUI
ENV COMFY_PORT=8188
ENV COMFY_VOLUME_ROOT=/workspace/ComfyUI
ENV COMFY_OUTPUT_DIR=/workspace/ComfyUI/output
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

RUN git clone https://github.com/comfyanonymous/ComfyUI.git "${COMFY_ROOT}"
RUN git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git "${COMFY_ROOT}/custom_nodes/ComfyUI-LTXVideo"
RUN git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git "${COMFY_ROOT}/custom_nodes/ComfyUI-VideoHelperSuite"

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install \
    --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
RUN python -m pip install -r "${COMFY_ROOT}/requirements.txt"
RUN python -m pip install \
    --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128 \
    --force-reinstall
RUN if [ -f "${COMFY_ROOT}/custom_nodes/ComfyUI-LTXVideo/requirements.txt" ]; then python -m pip install -r "${COMFY_ROOT}/custom_nodes/ComfyUI-LTXVideo/requirements.txt"; fi
RUN if [ -f "${COMFY_ROOT}/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt" ]; then python -m pip install -r "${COMFY_ROOT}/custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt"; fi
RUN python -m pip install runpod requests Pillow huggingface_hub

COPY handler.py /app/handler.py
COPY start.sh /app/start.sh
COPY workflows /app/workflows

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
