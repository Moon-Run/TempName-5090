#!/usr/bin/env bash
# Reuse existing Python/PyTorch/CUDA; fetch only the locked CUTLASS headers.
set -euo pipefail
setup_5090_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
setup_5090_python=${RTX5090_BASE_PYTHON:-${RTX5090_PYTHON:-python3}}
export PYTHONDONTWRITEBYTECODE=1
exec "$setup_5090_python" -B "$setup_5090_dir/setup_5090.py" "$@"
