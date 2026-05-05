#!/usr/bin/env bash
# =============================================================================
# install.sh — Set up SAM 3 for grounding benchmark evaluation
# =============================================================================
#
# Follows the SAM 3 setup from Facebook Research:
#   https://github.com/facebookresearch/sam3
#
#   1. Create conda env with PyTorch + CUDA
#   2. Clone SAM 3 and pip install -e .
#   3. Install barista-paper for dataset loading and evaluation
#
# Usage:
#   bash third_party/sam3/install.sh
#
# Requirements:
#   - conda (https://docs.conda.io/en/latest/miniconda.html)
#   - git
#   - Internet access
#
# After running:
#   conda activate sam3
#   python third_party/sam3/run_evaluation.py --help
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SAM3_SRC="${SCRIPT_DIR}/sam3_src"
CONDA_ENV="sam3"

echo "=== SAM 3 setup ==="
echo "Repo root : ${REPO_ROOT}"
echo "SAM 3 src : ${SAM3_SRC}"
echo "Conda env : ${CONDA_ENV}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Create conda environment
# ---------------------------------------------------------------------------
echo "[1/3] Creating conda environment..."

if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  Environment '${CONDA_ENV}' already exists."
    echo "  To recreate: conda env remove -n ${CONDA_ENV} && re-run this script."
else
    conda env create -f "${SCRIPT_DIR}/environment.yml" -n "${CONDA_ENV}"
    echo "  Environment created."
fi

# Install PyTorch with CUDA via pip. conda's pytorch-cuda channel often resolves
# to a CPU-only build when the system CUDA version doesn't match exactly.
# The cu121 wheel is compatible with CUDA 12.x drivers.
echo "  Installing PyTorch with CUDA support..."
conda run -n "${CONDA_ENV}" pip install \
    'torch>=2.5.1' \
    'torchvision>=0.20.1' \
    --index-url https://download.pytorch.org/whl/cu121

echo ""

# ---------------------------------------------------------------------------
# Step 2: Clone and install SAM 3
# ---------------------------------------------------------------------------
echo "[2/3] Cloning and installing SAM 3..."

if [[ -d "${SAM3_SRC}/.git" ]]; then
    echo "  SAM 3 already cloned, pulling latest..."
    git -C "${SAM3_SRC}" pull --ff-only
else
    git clone https://github.com/facebookresearch/sam3.git "${SAM3_SRC}"
fi

# Install sam3's runtime dependencies explicitly before the editable install.
# Upstream's package metadata currently omits triton and psutil even though
# they are imported at module import time in the default code path.
conda run -n "${CONDA_ENV}" pip install \
    "timm>=1.0.17" \
    "iopath>=0.1.10" \
    "ftfy==6.1.1" \
    "regex" \
    "huggingface_hub" \
    "tqdm" \
    "triton" \
    "psutil"
conda run -n "${CONDA_ENV}" pip install --no-deps -e "${SAM3_SRC}"

echo ""

# ---------------------------------------------------------------------------
# Step 3: Install barista-paper
# ---------------------------------------------------------------------------
echo "[3/3] Installing barista-paper (for dataset loading and evaluation)..."
echo "  (--no-deps to avoid overwriting torch from the conda env)"
conda run -n "${CONDA_ENV}" pip install --no-deps -e "${REPO_ROOT}"
conda run -n "${CONDA_ENV}" pip install \
    "setuptools<81" \
    "torchmetrics[detection]>=0.11" \
    "pycocotools>=2.0.7" \
    "pydantic>=2" \
    "pillow" \
    "numpy>=1.26,<2" \
    "opencv-python-headless" \
    "einops" \
    "imageio>=2.34" \
    "imageio-ffmpeg>=0.4.9"

echo ""

# ---------------------------------------------------------------------------
# Weights instructions
# ---------------------------------------------------------------------------
echo "Model weights"
echo ""
echo "  Download a SAM 3 checkpoint from the repository:"
echo "    https://github.com/facebookresearch/sam3#model-checkpoints"
echo ""
echo "  Save to: third_party/sam3/weights/"
echo "  Or pass --checkpoint <path> when running evaluation."
echo ""

# ---------------------------------------------------------------------------
# Verify setup
# ---------------------------------------------------------------------------
echo "Verifying setup..."
if ! conda run -n "${CONDA_ENV}" python -c '
import torch
print(f"  torch {torch.__version__} (CUDA: {torch.cuda.is_available()})")
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
print("  SAM 3: OK")
from barista.dataset import load_videos
print("  barista-paper: OK")
'; then
    echo "ERROR: Verification failed."
    exit 1
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "=== Setup complete ==="
echo ""
echo "  conda activate ${CONDA_ENV}"
echo ""
echo "  python third_party/sam3/run_evaluation.py \\"
echo "    --dataset-root <path/to/dataset> \\"
echo "    --benchmark-jsonl runs/vlm_datasets/grounding_data.jsonl \\"
echo "    --checkpoint third_party/sam3/weights/<checkpoint>.pt \\"
echo "    --out-dir runs/sam3"
echo ""
