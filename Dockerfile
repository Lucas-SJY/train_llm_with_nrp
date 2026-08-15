# CUDA 12.6 的 devel 镜像：DeepSpeed 需要 nvcc 做 JIT 编译，runtime 镜像会失败。
# 换 tag 前先确认 NRP 节点的驱动版本能带得动对应 CUDA。
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
# 训练数据（几十 MB）直接打进镜像，省掉一次往 PVC 拷数据的步骤。
# 数据量大了以后改成挂 PVC，见 README「数据放 PVC」一节。
COPY data/ data/

RUN chmod +x src/entrypoint.sh

CMD ["bash", "/workspace/src/entrypoint.sh"]
