#!/usr/bin/env bash
# Person-in-WiFi 3D training env for an RTX 4090 under WSL2 Ubuntu 22.04.
# Usage (inside WSL, repo checked out at $REPO):  bash csi_demo/train/setup_wsl.sh
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
ENV=piw3d

for tool in gcc g++ make git wget; do
  command -v "$tool" >/dev/null || { sudo apt-get update -y && sudo apt-get install -y build-essential git wget; break; }
done

if [ ! -d "$HOME/miniforge3" ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/mf.sh
  bash /tmp/mf.sh -b -p "$HOME/miniforge3"
fi
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda env list | grep -q "^$ENV " || conda create -y -n $ENV python=3.8
conda activate $ENV

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
conda install -y -c "nvidia/label/cuda-11.7.1" cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
export MAX_JOBS=${MAX_JOBS:-$(nproc)}

pip install "numpy<1.24" cython ninja
cd "$REPO/third_party/mmcv" && MMCV_WITH_OPS=1 pip install -e . -v
cd "$REPO/third_party/mmdet" && pip install -e .
cd "$REPO" && pip install -r requirements.txt && pip install -e .
pip install h5py scipy PyWavelets kaggle "numpy<1.24" yapf==0.32.0

python - <<'EOF'
import torch, mmcv, mmdet, opera
from mmcv.ops import get_compiling_cuda_version
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.get_device_name(0))
print("mmcv", mmcv.__version__, "ops cuda", get_compiling_cuda_version(), "| mmdet", mmdet.__version__)
from opera.datasets import WifiPoseCachedDataset
print("opera OK")
EOF
