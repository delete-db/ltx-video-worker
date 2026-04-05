FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the LTX-Video model during build (baked into image)
RUN python -c "from diffusers import LTXImageToVideoPipeline; LTXImageToVideoPipeline.from_pretrained('Lightricks/LTX-Video', torch_dtype='auto')" || echo "Model will download on first run"

# Copy handler
COPY handler.py .

# Start the serverless handler
CMD ["python", "-u", "handler.py"]
