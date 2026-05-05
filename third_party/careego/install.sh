#!/usr/bin/env bash
# =============================================================================
# install.sh — Set up CaRe-Ego for the hand_object benchmark evaluation
# =============================================================================
#
# Follows the official CaRe-Ego setup:
#   https://github.com/yuggiehk/CaRe-Ego
#
#   1. Clone CaRe-Ego and create conda env from their mmseg.yml
#   2. Clone MMSegmentation and pip install -e .
#   3. Copy CaRe-Ego files into MMSegmentation and patch __init__.py
#   4. python setup.py install
#   5. Install barista-paper for dataset loading and evaluation
#
# Usage:
#   bash third_party/careego/install.sh
#
# Requirements:
#   - conda (https://docs.conda.io/en/latest/miniconda.html)
#   - git
#   - Internet access
#
# After running:
#   conda activate careego
#   python third_party/careego/run_evaluation.py --help
# =============================================================================

set -euo pipefail

# Append import line to __init__.py if not already present
_append_import() {
    local file="$1"
    local line="$2"
    if ! grep -qF "${line}" "${file}"; then
        echo "${line}" >> "${file}"
        echo "    patched $(basename $(dirname "${file}"))/__init__.py"
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CAREEGO_SRC="${SCRIPT_DIR}/CaRe-Ego"
MMSEG_DIR="${SCRIPT_DIR}/mmsegmentation"
CONDA_ENV="careego"

echo "=== CaRe-Ego setup ==="
echo "Repo root   : ${REPO_ROOT}"
echo "CaRe-Ego src: ${CAREEGO_SRC}"
echo "MMSeg dir   : ${MMSEG_DIR}"
echo "Conda env   : ${CONDA_ENV}"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Clone CaRe-Ego and create conda environment
# ---------------------------------------------------------------------------
echo "[1/5] Cloning CaRe-Ego and creating conda environment..."

if [[ -d "${CAREEGO_SRC}/.git" ]]; then
    echo "  CaRe-Ego already cloned, pulling latest..."
    git -C "${CAREEGO_SRC}" pull --ff-only
else
    git clone https://github.com/yuggiehk/CaRe-Ego.git "${CAREEGO_SRC}"
fi

# CaRe-Ego provides mmseg.yml (or mmseg.yaml). Use it but override env name.
CAREEGO_YML="${CAREEGO_SRC}/mmseg.yml"
if [[ ! -f "${CAREEGO_YML}" ]]; then
    CAREEGO_YML="${CAREEGO_SRC}/mmseg.yaml"
fi
if [[ ! -f "${CAREEGO_YML}" ]]; then
    echo "ERROR: CaRe-Ego mmseg.yml not found. Check the CaRe-Ego repository."
    exit 1
fi

if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  Environment '${CONDA_ENV}' already exists."
    echo "  To recreate: conda env remove -n ${CONDA_ENV} && re-run this script."
else
    # mmcv==2.0.0 in mmseg.yaml must be built from source and requires CUDA headers
    # (cusolverDn.h from libcusolver-dev). We must install libcusolver-dev BEFORE
    # building mmcv. To do that, create the env without mmcv first, then install
    # libcusolver-dev, then build mmcv separately.
    CAREEGO_YML_NOMMCV="${CAREEGO_YML}.nommcv.yml"
    # Clean up temp file on exit (success or failure).
    trap 'rm -f "${CAREEGO_YML_NOMMCV}"' EXIT
    grep -v '^\s*- mmcv==' "${CAREEGO_YML}" > "${CAREEGO_YML_NOMMCV}"

    echo "  Step 1/2: Creating conda env (without mmcv)..."
    # SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL: CaRe-Ego pins sklearn==0.0.post5
    # which is deprecated; this env var allows pip to install it.
    SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
        conda env create -f "${CAREEGO_YML_NOMMCV}" -n "${CONDA_ENV}"
    echo "  Environment created (without mmcv)."
fi

# Install libcusolver-dev (cuSOLVER headers) BEFORE building mmcv from source.
# CaRe-Ego's mmseg.yaml has libcusolver (runtime) but not libcusolver-dev;
# cusolverDn.h from libcusolver-dev is required when compiling mmcv.
echo "  Installing libcusolver-dev (CUDA headers for mmcv build)..."
conda install -n "${CONDA_ENV}" -y -c nvidia libcusolver-dev --no-update-deps

# Now build and install mmcv (requires libcusolver-dev already present above).
# Pin numpy<2 explicitly: torch 1.x / torchvision 0.14 are compiled against numpy 1.x
# and break at runtime with numpy 2.x. Without the pin, pip may upgrade numpy when
# resolving mmcv's unconstrained numpy dependency.
echo "  Step 2/2: Building mmcv from source (C++ + CUDA compilation, this might take a while)..."
conda run -n "${CONDA_ENV}" pip install "mmcv==2.0.0" "numpy<2"

echo ""

# ---------------------------------------------------------------------------
# Step 2: Install MMSegmentation
# ---------------------------------------------------------------------------
echo "[2/5] Installing MMSegmentation..."

if [[ -d "${MMSEG_DIR}/.git" ]]; then
    echo "  MMSegmentation already cloned, pulling latest..."
    git -C "${MMSEG_DIR}" pull --ff-only
else
    git clone -b main https://github.com/open-mmlab/mmsegmentation.git "${MMSEG_DIR}"
fi

conda run -n "${CONDA_ENV}" pip install -v -e "${MMSEG_DIR}" "numpy<2"

echo ""

# ---------------------------------------------------------------------------
# Step 3: Copy CaRe-Ego files into MMSegmentation (per official README)
# ---------------------------------------------------------------------------
echo "[3/5] Patching CaRe-Ego into MMSegmentation..."

# Config
cp "${CAREEGO_SRC}/configs/CaRego.py" "${MMSEG_DIR}/configs/CaRego.py"
# Add test_pipeline (required by mmseg.apis.inference_model; not in original CaRe-Ego)
cat >> "${MMSEG_DIR}/configs/CaRego.py" << 'PATCHEOF'

# Required by mmseg.apis.inference_model (not in original CaRe-Ego config)
test_pipeline = [
    dict(type='LoadMultiLabelImageFromFile'),
    dict(type='ThreeLabelResizeSeperateTwoobj', scale=crop_size, keep_ratio=False),
    dict(type='PackSeperateTwoObjLabelSegInputs'),
]
PATCHEOF

# Dataset
cp "${CAREEGO_SRC}/datasets/EgoHOS_with_ORD.py" \
   "${MMSEG_DIR}/mmseg/datasets/EgoHOS_with_ORD.py"
_append_import "${MMSEG_DIR}/mmseg/datasets/__init__.py" \
    'from .EgoHOS_with_ORD import SeperateObjectEgohos'

# Data preprocessor
cp "${CAREEGO_SRC}/models/add_data_preprocess.py" \
   "${MMSEG_DIR}/mmseg/models/add_data_preprocess.py"
_append_import "${MMSEG_DIR}/mmseg/models/__init__.py" \
    'from .add_data_preprocess import SeperateTwoObjDataPreProcessor'

# Segmentor
cp "${CAREEGO_SRC}/models/segmentors/segmentor.py" \
   "${MMSEG_DIR}/mmseg/models/segmentors/segmentor.py"
_append_import "${MMSEG_DIR}/mmseg/models/segmentors/__init__.py" \
    'from .segmentor import CaregoSegmentor'

# Decoder heads (CaRe-Ego uses "deocder" typo in filenames)
for f in add_Unet_deocder_output.py add_unet_deocder_input.py \
         add_Unet_decoder_with_seperate_heads_obj.py; do
    cp "${CAREEGO_SRC}/models/decoder_heads/${f}" \
       "${MMSEG_DIR}/mmseg/models/decode_heads/${f}"
done
_append_import "${MMSEG_DIR}/mmseg/models/decode_heads/__init__.py" \
    'from .add_Unet_deocder_output import CaregoDecoder'
_append_import "${MMSEG_DIR}/mmseg/models/decode_heads/__init__.py" \
    'from .add_unet_deocder_input import CaregoDecoder2'
_append_import "${MMSEG_DIR}/mmseg/models/decode_heads/__init__.py" \
    'from .add_Unet_decoder_with_seperate_heads_obj import CaregoDecoder3'

# Transforms
cp "${CAREEGO_SRC}/datasets/transforms/add_transform_with_ORD.py" \
   "${MMSEG_DIR}/mmseg/datasets/transforms/add_transform_with_ORD.py"
cp "${CAREEGO_SRC}/datasets/transforms/add_transforms_egohos.py" \
   "${MMSEG_DIR}/mmseg/datasets/transforms/add_transforms_egohos.py"
_append_import "${MMSEG_DIR}/mmseg/datasets/transforms/__init__.py" \
    'from .add_transforms_egohos import LoadMultiLabelImageFromFile'
_append_import "${MMSEG_DIR}/mmseg/datasets/transforms/__init__.py" \
    'from .add_transform_with_ORD import LoadSeperateTwoObjAnnotation, LabelResizeSeperateTwoObj, RandomSeperateObjectCrop, PackSeperateTwoObjLabelSegInputs, ThreeLabelResizeSeperateTwoobj'

# Metrics
cp "${CAREEGO_SRC}/metrics/add_new_seperate_iou.py" \
   "${MMSEG_DIR}/mmseg/evaluation/metrics/add_new_seperate_iou.py"
_append_import "${MMSEG_DIR}/mmseg/evaluation/metrics/__init__.py" \
    'from .add_new_seperate_iou import NewSeperateIou'

# Rebuild per official README
conda run -n "${CONDA_ENV}" bash -c "cd '${MMSEG_DIR}' && python setup.py install"

echo ""

# ---------------------------------------------------------------------------
# Step 4: Install barista-paper
# ---------------------------------------------------------------------------
echo "[4/5] Installing barista-paper (for dataset loading and evaluation)..."
echo "  (--no-deps to avoid upgrading torch; CaRe-Ego needs torch 1.x from mmseg)"
conda run -n "${CONDA_ENV}" pip install --no-deps -e "${REPO_ROOT}"
echo "  Installing minimal deps compatible with CaRe-Ego's torch 1.x..."
conda run -n "${CONDA_ENV}" pip install "torchmetrics[detection]>=0.11,<1" "pycocotools>=2.0.7" "pydantic>=2" "ftfy" "regex" "einops" "imageio>=2.34" "imageio-ffmpeg>=0.4.9" "numpy<2"
# Install mmengine last: mmseg's setup.py install (step 3) and the pip installs above
# can displace it, so it must come after all other pip activity.
# --no-deps to avoid upgrading numpy past 1.x.
echo "  Installing mmengine..."
conda run -n "${CONDA_ENV}" pip install "mmengine==0.7.3" --no-deps

echo ""

# ---------------------------------------------------------------------------
# Step 5: Weights instructions
# ---------------------------------------------------------------------------
echo "[5/5] Model weights"
echo ""
echo "  Download the best mIoU checkpoint from the CaRe-Ego README:"
echo "    https://github.com/yuggiehk/CaRe-Ego#inference"
echo ""
echo "  Save to: third_party/careego/weights/best_mIoU_ckpt.pth"
echo "  Or pass --checkpoint <path> when running evaluation."
echo ""

# ---------------------------------------------------------------------------
# Verify setup
# ---------------------------------------------------------------------------
echo "Verifying setup..."
if ! conda run -n "${CONDA_ENV}" python -c '
from mmengine.config import Config
from mmengine.runner import Runner
from mmseg.apis import init_model
from barista.dataset import load_videos
print("  mmseg + barista-paper: OK")
'; then
    echo "ERROR: Verification failed. MMSegmentation or barista-paper is not importable."
    exit 1
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo "=== Setup complete ==="
echo ""
echo "  conda activate ${CONDA_ENV}"
echo ""
echo "  python third_party/careego/run_evaluation.py \\"
echo "    --dataset-root <path/to/dataset> \\"
echo "    --benchmark-jsonl runs/vlm_datasets/hand_object_data.jsonl \\"
echo "    --checkpoint third_party/careego/weights/best_mIoU_ckpt.pth \\"
echo "    --out-dir runs/careego"
echo ""
