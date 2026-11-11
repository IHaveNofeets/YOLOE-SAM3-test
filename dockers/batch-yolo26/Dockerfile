# syntax=docker/dockerfile:1
# Python packages come from the repo's wheels/ folder (compose passes it in as the "wheels" build context,
# tools/fetch_wheels.sh fills it), so a rebuild downloads nothing. Without it, this empty stage stands in
# and everything is downloaded.
FROM scratch AS wheels

FROM python:3.14-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 YOLO_CONFIG_DIR=/tmp/Ultralytics

# The package layers below (apt, pip) are the same as batch-yoloe's - same instructions, same
# requirements.txt - so Docker builds them once and both images share them on disk. (That is why the CLIP
# tokenizer, which only YOLOE's text prompts need, is installed here too.)
# libatomic1 / libnuma1: ROCm PyTorch links against them (the CPU/CUDA builds don't need them).
# libstdc++-14-dev: MIOpen compiles GPU kernels at run time, which needs the C++ headers.
RUN apt-get update && apt-get install -y --no-install-recommends libxcb1 libatomic1 libnuma1 libstdc++-14-dev \
    && rm -rf /var/lib/apt/lists/*

# cpu (default): CPU-only PyTorch + onnxruntime. cu130: CUDA 13 PyTorch + onnxruntime-gpu (also CUDA 13)
# for an NVIDIA GPU - set by compose.gpu.yaml. rocm7.2: ROCm PyTorch for an AMD GPU (training on the GPU,
# inference stays on onnxruntime's CPU build) - set by compose.amd.yaml.
ARG TORCH_VARIANT=cpu

COPY requirements.txt .
# inst: from wheels/ only, and only if something is missing there, from the internet.
# Headless OpenCV: ultralytics drags in the GUI build as a dependency, so it is swapped out at the end.
RUN --mount=type=bind,from=wheels,target=/wheels \
    --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    export PIP_FIND_LINKS=/wheels PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore \
    && inst() { PIP_NO_INDEX=1 pip install "$@" 2>/dev/null || pip install "$@"; } \
    && inst torch==2.13.0+${TORCH_VARIANT} torchvision==0.28.0+${TORCH_VARIANT} \
        --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && inst -r requirements.txt \
    && (PIP_NO_INDEX=1 pip install clip 2>/dev/null \
        || pip install "clip @ https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip") \
    && case "${TORCH_VARIANT}" in cu*) \
        pip uninstall -y onnxruntime && inst onnxruntime-gpu==1.29.0;; esac \
    && pip uninstall -y opencv-python opencv-python-headless \
    && inst opencv-python-headless==5.0.0.93

# The COCO-pretrained base weights are not baked into the image - they are downloaded on first use into this
# container's own dockers/batch-yolo26/modelData/yolo26 mount. Trained models persist in the models/ mount.
COPY detect_utils.py dataset.py evaluate.py train.py app.py ./
COPY static ./static

EXPOSE 7872
CMD ["python", "app.py"]
