#!/usr/bin/env bash
# One-time environment setup for a fresh qBraid GPU box: installs RAPIDS cudf.pandas
# (the GPU accelerator for pandas -- run_preprocessing.py uses plain `import pandas`
# throughout and gets GPU acceleration for free when launched via
# `python -m cudf.pandas run_preprocessing.py ...`), plus the rest of requirements.txt.
#
# Run from anywhere:  bash setup_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/../requirements.txt"

echo "=== GPU check ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found -- no GPU visible in this environment." >&2
    echo "The pipeline still runs correctly on CPU (plain pandas), just slower." >&2
    echo "Confirm you selected a GPU instance/kernel in qBraid before continuing." >&2
    CUDA_MAJOR=""
else
    nvidia-smi
    CUDA_VER=$(nvidia-smi | grep -oP "CUDA Version:\s*\K[0-9]+\.[0-9]+" || true)
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    echo "Detected driver CUDA version: ${CUDA_VER:-unknown}"
fi

echo
echo "=== Installing RAPIDS cudf.pandas ==="
if [ "$CUDA_MAJOR" = "12" ]; then
    PKG="cudf-cu12"
elif [ "$CUDA_MAJOR" = "11" ]; then
    PKG="cudf-cu11"
else
    echo "Could not confidently detect CUDA major version from nvidia-smi." >&2
    echo "Open https://docs.rapids.ai/install/ , pick your CUDA/Python/OS combo," >&2
    echo "copy the 'pip' command it gives you, run it manually, then re-run this" >&2
    echo "script's second half (the 'pip install -r requirements.txt' line below)." >&2
    PKG=""
fi

if [ -n "$PKG" ]; then
    # Version left unpinned deliberately: pin the exact release RAPIDS' own install
    # selector (https://docs.rapids.ai/install/) recommends for your CUDA/Python/OS
    # at the time you run this, rather than trusting a version baked in here that
    # may already be stale.
    pip install --extra-index-url=https://pypi.nvidia.com "$PKG"
    python -c "import cudf.pandas; print('cudf.pandas import OK')"
fi

echo
echo "=== Installing remaining requirements ==="
pip install -r "$REQ_FILE"

echo
echo "=== Verifying the pandas GPU proxy ==="
if [ -n "$PKG" ]; then
    python -m cudf.pandas -c "
import pandas as pd
print('pandas module ->', pd)
print('Is this the cudf.pandas proxy?', 'cudf' in str(type(pd)) or hasattr(pd, '_fsproxy_wrapped'))
"
else
    echo "Skipped (no RAPIDS package installed) -- pipeline will run on CPU pandas."
fi

echo
echo "Setup done. Run preprocessing with:"
echo "  python -m cudf.pandas ../src/run_preprocessing.py --input-dir <dataset> --output-dir <processed> --artifacts-dir <artifacts>"
echo "or, without GPU acceleration:"
echo "  python ../src/run_preprocessing.py --input-dir <dataset> --output-dir <processed> --artifacts-dir <artifacts>"
