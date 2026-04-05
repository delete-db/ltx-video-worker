FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

WORKDIR /app

# Install Python 3.12 + system deps
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Install uv (fast Python package manager)
RUN pip install uv

# Clone LTX-2 repo
RUN git clone https://github.com/Lightricks/LTX-2.git /app/ltx2

# Install LTX-2 dependencies
WORKDIR /app/ltx2
RUN uv sync

# Install runpod + extra deps into the LTX-2 venv using uv
RUN uv pip install --python /app/ltx2/.venv/bin/python runpod requests Pillow

# Copy our handler
COPY handler.py /app/handler.py

WORKDIR /app

# Run handler using the LTX-2 venv
CMD ["/app/ltx2/.venv/bin/python", "-u", "/app/handler.py"]
