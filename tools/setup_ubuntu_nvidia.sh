#!/usr/bin/env bash
# Set up an Ubuntu PC with an NVIDIA GPU to run the batch-yoloe and batch-sam3 containers on the GPU.
#
#   bash tools/setup_ubuntu_nvidia.sh --dataset /path/to/Dataset
#
# Run it from anywhere as your normal user (it asks for sudo when it needs it). It is safe to run again.
# What it does:
#   1. checks the system (Ubuntu 22.04 / 24.04, x86_64, an NVIDIA GPU)
#   2. installs Docker Engine + the compose plugin from Docker's own repository (if missing)
#   3. makes sure the NVIDIA driver is installed and new enough for CUDA 13 (>= 580); installs Ubuntu's
#      recommended driver if there is none (a reboot is needed after that - then run the script again)
#   4. installs the NVIDIA Container Toolkit (gives containers the GPU) and checks a container sees the GPU
#   5. adds you to the docker group
#   6. writes DATASET_ROOT into .env next to compose.yaml; checks that SAM 3's weights are
#      in dockers/batch-sam3/modelData/sam3 (copy sam3.pt there yourself)
#   7. builds and starts batch-yoloe (port 7870) and batch-sam3 (port 7871) with compose.gpu.yaml
#   8. checks that PyTorch (and batch-yoloe's ONNX Runtime) inside the containers can compute on the GPU
#
# Options:
#   --dataset PATH     the dataset root (has trainImages/, testImages/, refImages/); required the first time
#   --driver PACKAGE   install this driver package instead of Ubuntu's recommended one (e.g. nvidia-driver-580-open)
#   --no-start         set everything up but don't build / start the containers
#   -h, --help         this text
set -euo pipefail

# ----------------------------------------------------------------------------- helpers
bold() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*" >&2; }
die() { printf '\033[31mxx %s\033[0m\n' "$*" >&2; exit 1; }

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0; }

MIN_DRIVER=580  # CUDA 13 (the cu130 PyTorch and onnxruntime-gpu in the images) needs driver 580 or newer
DATASET=""
DRIVER_PKG=""
START=1
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --dataset=*) DATASET="${1#*=}"; shift ;;
    --driver) DRIVER_PKG="${2:-}"; shift 2 ;;
    --driver=*) DRIVER_PKG="${1#*=}"; shift ;;
    --no-start) START=0; shift ;;
    -h|--help) usage ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

[ "$(id -u)" -ne 0 ] || die "run this as your normal user, not root (it uses sudo where needed)"
command -v sudo >/dev/null || die "sudo is needed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$REPO/compose.yaml" ] && [ -f "$REPO/compose.gpu.yaml" ] || die "can't find compose.yaml / compose.gpu.yaml in $REPO"
ENV_FILE="$REPO/.env"
NEED_RELOGIN=0
NEED_REBOOT=0

# set KEY=VALUE in .env, keeping every other line
set_env() {
  local key="$1" value="$2"
  touch "$ENV_FILE"
  if grep -q "^${key}=" "$ENV_FILE"; then
    sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
  else
    printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
  fi
}
get_env() { [ -f "$ENV_FILE" ] && grep "^$1=" "$ENV_FILE" | tail -1 | cut -d= -f2- || true; }

# the installed driver's version (e.g. 580.95.05), empty if nvidia-smi doesn't work
driver_version() { nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true; }

# ----------------------------------------------------------------------------- 1. system
bold "1/8  Checking the system"
[ -r /etc/os-release ] || die "not a Linux with /etc/os-release"
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || die "this script is for Ubuntu (found ${ID:-unknown})"
case "${VERSION_ID:-}" in
  22.04|24.04) ;;
  *) warn "tested for Ubuntu 22.04 / 24.04, this is ${VERSION_ID:-?} - continuing" ;;
esac
CODENAME="${VERSION_CODENAME:-${UBUNTU_CODENAME:-}}"
[ "$(uname -m)" = "x86_64" ] || die "x86_64 only"
info "Ubuntu ${VERSION_ID} (${CODENAME}), kernel $(uname -r)"

sudo apt-get update -qq
sudo apt-get install -y -qq ca-certificates curl gnupg pciutils >/dev/null
GPU_LINE="$(lspci -nn | grep -Ei 'vga|display|3d' | grep -i nvidia | head -1 || true)"
[ -n "$GPU_LINE" ] || die "no NVIDIA GPU found by lspci"
info "GPU: ${GPU_LINE#*: }"

# ----------------------------------------------------------------------------- 2. docker
bold "2/8  Docker Engine"
if command -v docker >/dev/null && docker compose version >/dev/null 2>&1; then
  info "already installed: $(docker --version)"
else
  if command -v docker >/dev/null; then
    warn "docker is installed but without the compose plugin; installing Docker's own packages"
  fi
  # Docker's official apt repository (https://docs.docker.com/engine/install/ubuntu/)
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${CODENAME} stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin >/dev/null
  sudo systemctl enable --now docker
  info "installed: $(docker --version)"
fi

# ----------------------------------------------------------------------------- 3. NVIDIA driver
bold "3/8  NVIDIA driver (CUDA 13 needs >= ${MIN_DRIVER})"
install_driver() {
  sudo apt-get install -y -qq ubuntu-drivers-common >/dev/null
  if command -v mokutil >/dev/null && mokutil --sb-state 2>/dev/null | grep -qi enabled; then
    warn "Secure Boot is on: if you are asked for a MOK password, set one, and choose 'Enroll MOK' at the next boot"
  fi
  if [ -n "$DRIVER_PKG" ]; then
    info "installing $DRIVER_PKG"
    sudo apt-get install -y "$DRIVER_PKG"
  else
    info "installing Ubuntu's recommended driver: $(ubuntu-drivers devices 2>/dev/null | grep -o 'nvidia-driver-[^ ]*.*recommended' | head -1 || echo '?')"
    sudo ubuntu-drivers install
  fi
  NEED_REBOOT=1
}
DRV="$(driver_version)"
if [ -n "$DRIVER_PKG" ] && ! dpkg -s "$DRIVER_PKG" >/dev/null 2>&1; then
  install_driver
elif [ -z "$DRV" ]; then
  if dpkg -l 'nvidia-driver-*' 2>/dev/null | grep -q '^ii'; then
    warn "an NVIDIA driver is installed but nvidia-smi doesn't work: reboot first (then run this script again)"
    NEED_REBOOT=1
  else
    warn "no NVIDIA driver: installing one"
    install_driver
  fi
elif [ "${DRV%%.*}" -lt "$MIN_DRIVER" ]; then
  die "driver $DRV is too old for CUDA 13 (needs >= ${MIN_DRIVER}). Install a newer one, e.g.:
   bash tools/setup_ubuntu_nvidia.sh --driver nvidia-driver-${MIN_DRIVER}-open   (or nvidia-driver-${MIN_DRIVER}; see 'ubuntu-drivers devices')"
else
  info "driver $DRV: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
fi

# ----------------------------------------------------------------------------- 4. container toolkit
bold "4/8  NVIDIA Container Toolkit"
if command -v nvidia-ctk >/dev/null; then
  info "already installed: $(nvidia-ctk --version 2>/dev/null | head -1)"
else
  # NVIDIA's apt repository (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq nvidia-container-toolkit >/dev/null
  info "installed: $(nvidia-ctk --version 2>/dev/null | head -1)"
fi
if ! sudo docker info 2>/dev/null | grep -qi 'runtimes:.*nvidia'; then
  sudo nvidia-ctk runtime configure --runtime=docker >/dev/null
  sudo systemctl restart docker
  info "registered the NVIDIA runtime with Docker"
fi

# ----------------------------------------------------------------------------- 5. groups
bold "5/8  Group (docker)"
if id -nG "$USER" | tr ' ' '\n' | grep -qx docker; then
  info "$USER is in docker"
else
  sudo usermod -aG docker "$USER"
  info "added $USER to docker"
  NEED_RELOGIN=1
fi

# ----------------------------------------------------------------------------- 6. .env + weights
bold "6/8  Settings in $ENV_FILE"
[ -n "$DATASET" ] || DATASET="$(get_env DATASET_ROOT)"
[ -n "$DATASET" ] || die "give the dataset root with --dataset /path/to/Dataset"
DATASET="$(cd "$DATASET" 2>/dev/null && pwd)" || die "dataset folder not found: $DATASET"
[ -d "$DATASET/trainImages" ] || [ -d "$DATASET/testImages" ] || die "$DATASET has no trainImages/ or testImages/"
for out in YOLOE SAM3; do
  mkdir -p "$DATASET/output/$out" 2>/dev/null || true
  if ! touch "$DATASET/output/$out/.write_test" 2>/dev/null; then
    die "can't write to $DATASET/output/$out. If the dataset is on the Windows (NTFS) drive: turn off Windows Fast Startup
   and shut Windows down fully, then mount the drive read-write (e.g. in /etc/fstab with the ntfs3 driver, uid=$(id -u),gid=$(id -g))"
  fi
  rm -f "$DATASET/output/$out/.write_test"
done
set_env DATASET_ROOT "$DATASET"
info "DATASET_ROOT=$DATASET"

# SAM 3's weights: batch-sam3 has its own copy (the download is gated on Hugging Face, so nothing fetches it)
SAM3_PT="$REPO/dockers/batch-sam3/modelData/sam3/sam3.pt"
SAM3_OK=1
if [ -f "$SAM3_PT" ]; then
  info "SAM 3 weights: $SAM3_PT"
else
  SAM3_OK=0
  warn "SAM 3 weights not found: put sam3.pt in $(dirname "$SAM3_PT")/ (batch-sam3 starts, but can't run until then)"
fi

# ----------------------------------------------------------------------------- 7. build + start
SERVICES=(batch-yoloe batch-sam3)
COMPOSE=(sudo docker compose --project-directory "$REPO" -f "$REPO/compose.yaml" -f "$REPO/compose.gpu.yaml")
if [ "$START" = 0 ]; then
  bold "7/8  Skipped (--no-start). Start later with:"
  info "cd $REPO && docker compose -f compose.yaml -f compose.gpu.yaml up -d --build ${SERVICES[*]}"
else
  if [ "$NEED_REBOOT" = 1 ]; then
    bold "7/8  Reboot needed first"
    info "The NVIDIA driver was installed. Reboot, then run this script again (same command) to build and start."
    exit 0
  fi
  info "checking that a container sees the GPU..."
  sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L | sed 's/^/   container: /' \
    || die "a container can't use the GPU; check 'nvidia-smi' on the host and 'sudo nvidia-ctk runtime configure --runtime=docker'"
  bold "7/8  Building and starting ${SERVICES[*]} (the first build downloads several GB of CUDA PyTorch)"
  "${COMPOSE[@]}" up -d --build "${SERVICES[@]}"
  info "waiting for them to answer..."
  for port in 7870 7871; do
    # any HTTP answer counts as started (batch-sam3's /health says 503 while its weights are missing)
    for _ in $(seq 1 120); do
      curl -s -o /dev/null http://localhost:$port/health && break
      sleep 3
    done
    curl -s -o /dev/null http://localhost:$port/health \
      || die "the container on port $port did not start; see: ${COMPOSE[*]} logs ${SERVICES[*]}"
  done

  # --------------------------------------------------------------------------- 8. check the GPU
  bold "8/8  Checking the GPU inside the containers"
  for svc in "${SERVICES[@]}"; do
    info "$svc:"
    if "${COMPOSE[@]}" exec -T "$svc" python -c "
import sys, torch
ok = torch.cuda.is_available()
print('   CUDA PyTorch:', torch.__version__, '| CUDA', torch.version.cuda, '| GPU available:', ok)
if not ok:
    sys.exit(1)
print('   GPU:', torch.cuda.get_device_name(0), '| memory %.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 1e9))
a = torch.randn(2048, 2048, device='cuda')
b = (a @ a).sum().item()
torch.cuda.synchronize()
print('   a test computation on the GPU worked')
try:
    import onnxruntime  # batch-yoloe only: its tests run the models through ONNX Runtime
except ImportError:
    sys.exit(0)
gpu = 'CUDAExecutionProvider' in onnxruntime.get_available_providers()
print('   ONNX Runtime', onnxruntime.__version__, '| CUDA provider:', gpu)
sys.exit(0 if gpu else 2)
"; then
      info "OK"
    else
      warn "$svc can't fully use the GPU. Check 'nvidia-smi' (driver >= ${MIN_DRIVER}) and the container log"
      warn "(${COMPOSE[*]} logs $svc). It runs on the CPU meanwhile."
    fi
  done
  for port in 7870 7871; do
    curl -s http://localhost:$port/api/models | grep -o '"device":"[^"]*"' | sed "s/^/   :$port reports /" || true
  done
  info "open http://localhost:7870 (YOLOE) and http://localhost:7871 (SAM 3)"
  [ "$SAM3_OK" = 1 ] || warn "batch-sam3 still needs sam3.pt in $(dirname "$SAM3_PT")/"
fi

if [ "$NEED_RELOGIN" = 1 ]; then
  bold "Log out and back in (or reboot)"
  info "so your new group membership applies; after that 'docker' works without sudo."
fi
bold "Done"
info "Start / update later:  cd $REPO && docker compose -f compose.yaml -f compose.gpu.yaml up -d --build ${SERVICES[*]}"
