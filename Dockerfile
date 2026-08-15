# CUDA 12.6 devel image: DeepSpeed needs nvcc for its JIT ops, the runtime image fails.
# Before changing the tag, check that the NRP node drivers support the matching CUDA version.
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/
# The training data (tens of MB) is baked into the image, which saves one copy
# step to the PVC. Switch to a PVC mount once the dataset grows; see README.
COPY data/ data/

RUN chmod +x src/entrypoint.sh

CMD ["bash", "/workspace/src/entrypoint.sh"]
