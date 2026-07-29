#!/usr/bin/env bash

set -euo pipefail

# Config
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_VERSION="${CUDA_VERSION:-12.4}"

SUDO=""; if [ "${EUID:-$(id -u)}" -ne 0 ]; then SUDO="sudo"; fi

# Sys deps
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update
$SUDO apt-get install -y \
  git git-lfs curl build-essential cmake ninja-build pkg-config python3-dev \
  libgl1-mesa-dev libglfw3 libglfw3-dev libglew-dev xorg-dev \
  libxi-dev libxinerama-dev libxcursor1 libxrandr2 ffmpeg htop \
  python3-venv python3-pip
git lfs install || true

# uv
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Configure git to skip LFS during dependency installations
export GIT_LFS_SKIP_SMUDGE=1
git config --global filter.lfs.smudge "git-lfs smudge --skip %f || cat"
git config --global filter.lfs.process "git-lfs filter-process --skip || cat"
git config --global lfs.fetchinclude ""
git config --global lfs.fetchexclude "*"

# Install everything with uv sync (creates lock file, installs all deps including OpenPI)
echo "Installing B1K solution package and all dependencies (including OpenPI)..."
echo "This creates a lock file and ensures consistent installations."
uv sync --extra dev

# BEHAVIOR-1K v3.9.1 uses a newer LeRobot dependency than OpenPI. Keep its
# simulator dependencies in the separate `behavior` Conda environment described
# in docs/INSTALLATION.md instead of mutating this locked training environment.

# Setup Jupyter kernel for development
echo "Setting up Jupyter kernel..."
uv run python -m ipykernel install --user --name=b1k --display-name "Python (B1K)"

# Fix for `/usr/bin/ld: cannot find -lcuda:` error
ldconfig -p | grep libcuda || true
ls -l /usr/lib/x86_64-linux-gnu/libcuda.so* /lib/x86_64-linux-gnu/libcuda.so* || true

# Create missing unversioned .so
$SUDO ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so || true
$SUDO ln -sf /lib/x86_64-linux-gnu/libcuda.so.1 /lib/x86_64-linux-gnu/libcuda.so || true

# Make sure loader sees it
$SUDO ldconfig
