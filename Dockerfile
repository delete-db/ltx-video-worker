FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python and system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-venv \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install PyTorch 2.11.0 with CUDA 12.8
RUN pip install --no-cache-dir \
    torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

CMD ["python", "-u", "handler.py"]
