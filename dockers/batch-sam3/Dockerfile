# syntax=docker/dockerfile:1
FROM python:3.14-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# cpu (default): CPU-only PyTorch. cu130: CUDA 13 PyTorch for an NVIDIA GPU - set by compose.gpu.yaml.
# rocm7.2: ROCm PyTorch for an AMD GPU - set by compose.amd.yaml. SAM 3 runs in PyTorch, so a GPU build
# uses the GPU for inference and prompt tuning. PyPI's default Linux torch is the CUDA build, so torch
# always comes from PyTorch's own index for the chosen variant, before the rest.
ARG TORCH_VARIANT=cpu

COPY requirements.txt .
# Same headless-OpenCV dance as dockers/yoloe: ultralytics drags in the GUI build as a dependency.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.13.0+${TORCH_VARIANT} torchvision==0.28.0+${TORCH_VARIANT} \
        --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip install -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install opencv-python-headless==5.0.0.93

# ROCm PyTorch links against these system libraries (the CPU/CUDA builds don't need them), and MIOpen
# compiles GPU kernels at run time, which needs the C++ standard library headers. After the pip layer so
# adding them doesn't redo the big torch download.
RUN apt-get update && apt-get install -y --no-install-recommends libatomic1 libnuma1 libstdc++-14-dev \
    && rm -rf /var/lib/apt/lists/*

# The SAM 3 weights are not baked into the image: they are in this container's own
# dockers/batch-sam3/modelData/sam3 mount. Built models persist in the models/ mount.
COPY detect_utils.py dataset.py evaluate.py sam3_models.py prompt_tune.py app.py ./
COPY static ./static

EXPOSE 7871
CMD ["python", "app.py"]
