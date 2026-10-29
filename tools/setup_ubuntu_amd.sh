#!/usr/bin/env bash
# Set up an Ubuntu PC with an AMD Radeon GPU to run the YOLOE batch-yoloe container with GPU training.
#
#   bash tools/setup_ubuntu_amd.sh --dataset /path/to/Dataset
#
# Run it from anywhere as your normal user (it asks for sudo when it needs it). It is safe to run again.
# What it does:
#   1. checks the system (Ubuntu 22.04 / 24.04, x86_64, an AMD GPU)
#   2. installs Docker Engine + the compose plugin from Docker's own repository (if missing)
#   3. makes sure the AMD kernel driver exposes ROCm (/dev/kfd); installs AMD's amdgpu-dkms only if not
#   4. adds you to the docker / render / video groups
#   5. writes .env next to compose.yaml: DATASET_ROOT, the render / video group ids, and
#      HSA_OVERRIDE_GFX_VERSION when the GPU needs it (e.g. RX 6750 XT = gfx1031 -> 10.3.0)
#   6. builds and starts the batch-yoloe container with compose.amd.yaml (ROCm PyTorch)
#   7. checks that PyTorch inside the container sees the GPU and can compute on it
#
# Options:
#   --dataset PATH     the dataset root (has trainImages/, testImages/, refImages/); required the first time
#   --amdgpu-dkms      install AMD's amdgpu-dkms driver even if the built-in kernel driver works
#   --no-start         set everything up but don't build / start the container
#   -h, --help         this text
set -euo pipefail

# ----------------------------------------------------------------------------- helpers
bold() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
info() { printf '   %s\n' "$*"; }
warn() { printf '\033[33m!! %s\033[0m\n' "$*" >&2; }
die() { printf '\033[31mxx %s\033[0m\n' "$*" >&2; exit 1; }

usage() { sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

DATASET=""
FORCE_DKMS=0
START=1
while [ $# -gt 0 ]; do
  case "$1" in
    --dataset) DATASET="${2:-}"; shift 2 ;;
    --dataset=*) DATASET="${1#*=}"; shift ;;
    --amdgpu-dkms) FORCE_DKMS=1; shift ;;
    --no-start) START=0; shift ;;
    -h|--help) usage ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

[ "$(id -u)" -ne 0 ] || die "run this as your normal user, not root (it uses sudo where needed)"
command -v sudo >/dev/null || die "sudo is needed"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
[ -f "$REPO/compose.yaml" ] && [ -f "$REPO/compose.amd.yaml" ] || die "can't find compose.yaml / compose.amd.yaml in $REPO"
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

# ----------------------------------------------------------------------------- 1. system
bold "1/7  Checking the system"
[ -r /etc/os-release ] || die "not a Linux with /etc/os-release"
. /etc/os-release
[ "${ID:-}" = "ubuntu" ] || die "this script is for Ubuntu (found ${ID:-unknown})"
case "${VERSION_ID:-}" in
  22.04|24.04) ;;
  *) warn "tested for Ubuntu 22.04 / 24.04, this is ${VERSION_ID:-?} - continuing" ;;
esac
CODENAME="${VERSION_CODENAME:-${UBUNTU_CODENAME:-}}"
[ "$(uname -m)" = "x86_64" ] || die "x86_64 only (ROCm PyTorch has no other builds)"
info "Ubuntu ${VERSION_ID} (${CODENAME}), kernel $(uname -r)"

sudo apt-get update -qq
sudo apt-get install -y -qq ca-certificates curl gnupg pciutils >/dev/null
GPU_LINE="$(lspci -nn | grep -Ei 'vga|display|3d' | grep -Ei 'amd|ati|radeon' | head -1 || true)"
[ -n "$GPU_LINE" ] || die "no AMD GPU found by lspci"
info "GPU: ${GPU_LINE#*: }"

# ----------------------------------------------------------------------------- 2. docker
bold "2/7  Docker Engine"
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

# ----------------------------------------------------------------------------- 3. AMD driver
bold "3/7  AMD kernel driver (ROCm needs /dev/kfd)"
install_dkms() {
  # AMD's installer package for this Ubuntu release, from the "latest" folder of AMD's repository
  local base="https://repo.radeon.com/amdgpu-install/latest/ubuntu/${CODENAME}/"
  local deb
  deb="$(curl -fsSL "$base" | grep -o 'amdgpu-install_[^"<>]*_all\.deb' | sort -V | tail -1 || true)"
  [ -n "$deb" ] || die "could not find amdgpu-install for ${CODENAME} at $base"
  info "installing AMD driver package $deb (this takes a while: it builds a kernel module)"
  curl -fsSL -o "/tmp/$deb" "$base$deb"
  sudo apt-get install -y -qq "/tmp/$deb" >/dev/null
  sudo apt-get update -qq
  # only the kernel driver: ROCm itself (the user-space part) comes inside the container with PyTorch
  sudo amdgpu-install -y --usecase=dkms --no-32
  NEED_REBOOT=1
}
if [ "$FORCE_DKMS" = 1 ]; then
  install_dkms
elif [ -e /dev/kfd ]; then
  info "/dev/kfd exists: the built-in amdgpu driver provides ROCm access (use --amdgpu-dkms to install AMD's driver anyway)"
else
  warn "/dev/kfd is missing: installing AMD's amdgpu-dkms driver"
  install_dkms
fi

# ----------------------------------------------------------------------------- 4. groups
bold "4/7  Groups (docker, render, video)"
for g in docker render video; do
  if ! getent group "$g" >/dev/null; then
    sudo groupadd -f "$g"
  fi
  if id -nG "$USER" | tr ' ' '\n' | grep -qx "$g"; then
    info "$USER is in $g"
  else
    sudo usermod -aG "$g" "$USER"
    info "added $USER to $g"
    NEED_RELOGIN=1
  fi
done
RENDER_GID="$(getent group render | cut -d: -f3)"
VIDEO_GID="$(getent group video | cut -d: -f3)"

# ----------------------------------------------------------------------------- 5. .env
bold "5/7  Settings in $ENV_FILE"
[ -n "$DATASET" ] || DATASET="$(get_env DATASET_ROOT)"
[ -n "$DATASET" ] || die "give the dataset root with --dataset /path/to/Dataset"
DATASET="$(cd "$DATASET" 2>/dev/null && pwd)" || die "dataset folder not found: $DATASET"
[ -d "$DATASET/trainImages" ] || [ -d "$DATASET/testImages" ] || die "$DATASET has no trainImages/ or testImages/"
mkdir -p "$DATASET/output/YOLOE" 2>/dev/null || true
if ! touch "$DATASET/output/YOLOE/.write_test" 2>/dev/null; then
  die "can't write to $DATASET/output/YOLOE. If the dataset is on the Windows (NTFS) drive: turn off Windows Fast Startup
   and shut Windows down fully, then mount the drive read-write (e.g. in /etc/fstab with the ntfs3 driver, uid=$(id -u),gid=$(id -g))"
fi
rm -f "$DATASET/output/YOLOE/.write_test"

# which gfx target the GPU is; RDNA2 cards other than gfx1030 (RX 6800/6900) run as gfx1030
GFX=""
for p in /sys/class/kfd/kfd/topology/nodes/*/properties; do
  v="$(awk '$1=="gfx_target_version"{print $2}' "$p" 2>/dev/null || true)"
  if [ -n "$v" ] && [ "$v" != "0" ]; then GFX="$v"; break; fi
done
OVERRIDE=""
if [ -n "$GFX" ]; then
  major=$((GFX / 10000)); minor=$(((GFX / 100) % 100)); step=$((GFX % 100))
  info "GPU target: gfx${major}${minor}$(printf '%x' "$step") (gfx_target_version $GFX)"
  if [ "$major" = 10 ] && [ "$minor" = 3 ] && [ "$step" != 0 ]; then
    OVERRIDE="10.3.0"
    info "not on ROCm's official list: it will run as gfx1030 (HSA_OVERRIDE_GFX_VERSION=10.3.0)"
  fi
else
  warn "could not read the GPU target from /sys/class/kfd (after a reboot it should be there); assuming an RX 6000 card"
  OVERRIDE="10.3.0"
fi
set_env DATASET_ROOT "$DATASET"
set_env RENDER_GID "$RENDER_GID"
set_env VIDEO_GID "$VIDEO_GID"
set_env HSA_OVERRIDE_GFX_VERSION "$OVERRIDE"
info "DATASET_ROOT=$DATASET  RENDER_GID=$RENDER_GID  VIDEO_GID=$VIDEO_GID  HSA_OVERRIDE_GFX_VERSION=${OVERRIDE:-(none)}"

# ----------------------------------------------------------------------------- 6. build + start
COMPOSE=(sudo docker compose --project-directory "$REPO" -f "$REPO/compose.yaml" -f "$REPO/compose.amd.yaml")
if [ "$START" = 0 ]; then
  bold "6/7  Skipped (--no-start). Start later with:"
  info "cd $REPO && docker compose -f compose.yaml -f compose.amd.yaml up -d --build batch-yoloe"
else
  if [ "$NEED_REBOOT" = 1 ]; then
    bold "6/7  Reboot needed first"
    info "The AMD driver was installed. Reboot, then run this script again (same command) to build and start."
    exit 0
  fi
  [ -e /dev/kfd ] || die "/dev/kfd still missing - reboot, or run again with --amdgpu-dkms"
  bold "6/7  Building and starting the batch-yoloe container (the first build downloads ~5 GB of ROCm PyTorch)"
  "${COMPOSE[@]}" up -d --build batch-yoloe
  info "waiting for it to answer..."
  for _ in $(seq 1 120); do
    curl -sf http://localhost:7870/health >/dev/null && break
    sleep 3
  done
  curl -sf http://localhost:7870/health >/dev/null || die "the container did not start; see: ${COMPOSE[*]} logs batch-yoloe"

  # --------------------------------------------------------------------------- 7. check the GPU
  bold "7/7  Checking the GPU inside the container"
  if "${COMPOSE[@]}" exec -T batch-yoloe python -c "
import torch, sys
ok = torch.cuda.is_available()
print('   ROCm PyTorch:', torch.__version__, '| HIP', torch.version.hip, '| GPU available:', ok)
if not ok:
    sys.exit(1)
print('   GPU:', torch.cuda.get_device_name(0), '| memory %.1f GB' % (torch.cuda.get_device_properties(0).total_memory / 1e9))
a = torch.randn(2048, 2048, device='cuda')
b = (a @ a).sum().item()
torch.cuda.synchronize()
print('   a test computation on the GPU worked')
"; then
    info "OK"
  else
    warn "PyTorch in the container can't use the GPU. Check: 'ls -l /dev/kfd /dev/dri', the HSA_OVERRIDE_GFX_VERSION"
    warn "in $ENV_FILE, and the container log (${COMPOSE[*]} logs batch-yoloe). Training will run on the CPU meanwhile."
  fi
  curl -s http://localhost:7870/api/models | grep -o '"device":"[^"]*"' | sed 's/^/   page reports /' || true
  info "open http://localhost:7870"
fi

if [ "$NEED_RELOGIN" = 1 ]; then
  bold "Log out and back in (or reboot)"
  info "so your new group memberships apply; after that 'docker' works without sudo."
fi
bold "Done"
info "Start / update later:  cd $REPO && docker compose -f compose.yaml -f compose.amd.yaml up -d --build batch-yoloe"
