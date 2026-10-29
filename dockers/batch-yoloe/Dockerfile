# syntax=docker/dockerfile:1
FROM python:3.14-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# cpu (default): CPU-only PyTorch + onnxruntime. cu130: CUDA 13 PyTorch + onnxruntime-gpu (also CUDA 13)
# for an NVIDIA GPU - set by compose.gpu.yaml. rocm7.2: ROCm PyTorch for an AMD GPU (training on the GPU,
# inference stays on onnxruntime's CPU build) - set by compose.amd.yaml. PyPI's default Linux torch is the
# CUDA build, so torch always comes from PyTorch's own index for the chosen variant, before the rest.
ARG TORCH_VARIANT=cpu

COPY requirements.txt .
# Headless OpenCV: ultralytics drags in the GUI build as a dependency, so it is swapped out at the end.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0+${TORCH_VARIANT} torchvision==0.28.0+${TORCH_VARIANT} \
        --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip install -r requirements.txt \
    && case "${TORCH_VARIANT}" in cu*) \
        pip uninstall -y onnxruntime && pip install onnxruntime-gpu==1.29.0;; esac \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install opencv-python-headless==5.0.0.93

# ROCm PyTorch links against these system libraries (the CPU/CUDA builds don't need them), and MIOpen
# compiles GPU kernels at run time, which needs the C++ standard library headers. After the pip layer so
# adding them doesn't redo the big torch download.
RUN apt-get update && apt-get install -y --no-install-recommends libatomic1 libnuma1 libstdc++-14-dev \
    && rm -rf /var/lib/apt/lists/*

# Base weights are not baked into the image - app.py downloads them on first start into this
# container's own dockers/batch-yoloe/modelData/yoloe mount. Baked models persist in the models/ mount.
COPY yoloe_bake.py detect_utils.py dataset.py evaluate.py train.py app.py ./
COPY static ./static

EXPOSE 7870
CMD ["python", "app.py"]
