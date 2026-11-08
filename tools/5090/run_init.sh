#!/usr/bin/env bash
set -euo pipefail
rtx_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
rtx_python=${RTX5090_PYTHON:-python3}
export PYTHONDONTWRITEBYTECODE=1
cd -- "$rtx_repo"
exec "$rtx_python" tools/5090/init.py "$@"
