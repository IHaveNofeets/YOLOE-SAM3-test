#!/bin/sh
# Downloads every Python package (wheel) that batch-yoloe and batch-sam3 need into the repo's wheels/
# folder, for one PyTorch build. The Dockerfiles install from that folder, so rebuilding the images
# downloads nothing. Runs in a container through compose (TORCH_VARIANT comes from the compose files):
#
#   docker compose run --rm wheels                                        # CPU
#   docker compose -f compose.yaml -f compose.gpu.yaml run --rm wheels    # NVIDIA (cu130)
#   docker compose -f compose.yaml -f compose.amd.yaml run --rm wheels    # AMD (rocm7.2)
#
# Files already in wheels/ are never downloaded again, and when nothing is missing it doesn't go online
# at all. Run it again after changing a requirements.txt: only the new packages are downloaded.
set -e
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
V="${TORCH_VARIANT:-cpu}"
TORCH="torch==2.13.0+$V torchvision==0.28.0+$V"
CLIP_URL=https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip
cd /tmp
# ultralytics asks for "torch>=...": this keeps pip on the chosen build instead of PyPI's own torch
echo "$TORCH" | tr ' ' '\n' > constraints.txt

# offline first (wheels/ only); online, for whatever is missing, only if that fails
fetch() {
  PIP_NO_INDEX=1 pip download -q -d /wheels --find-links /wheels "$@" 2>/dev/null \
    || pip download -d /wheels --find-links /wheels "$@"
}

echo "== PyTorch ($V)"
# shellcheck disable=SC2086
fetch -c constraints.txt $TORCH --index-url "https://download.pytorch.org/whl/$V"

echo "== CLIP tokenizer (built once from GitHub)"
ls /wheels/clip-*.whl >/dev/null 2>&1 || pip wheel -q --no-deps -w /wheels "clip @ $CLIP_URL"

echo "== the rest"
EXTRA=""
case "$V" in cu*) EXTRA="onnxruntime-gpu==1.29.0";; esac
# shellcheck disable=SC2086
fetch -c constraints.txt -r /src/batch-yoloe/requirements.txt -r /src/batch-sam3/requirements.txt clip $EXTRA

echo "done: $(ls /wheels | wc -l) files, $(du -sh /wheels | cut -f1) in wheels/"
