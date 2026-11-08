#!/usr/bin/env bash
# Run inside the Ubuntu22.04 CUDA12.8 container. No training is launched.
set -euo pipefail
setup_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

for arg in "$@"; do
    case "$arg" in
        --help|-h)
            cat <<'HELP'
Usage: bash env/setup_h100.sh [options]
Install Python3.10.20, PyTorch2.6.0+cu124, CUDA12.4/12.8, Apex and pinned sources.
Options:
  --prefix PATH       Training environment (default: ../conda-envs/tempname-h100)
  --megatron PATH     Megatron checkout (default: ../Megatron-LM)
  --state-dir PATH    Bootstrap/cache directory (default: ../.h100-env)
  --skip-gpu-check    Configure software without GPU identity/P2P validation
  --dry-run          Print the plan only; no installs, writes or GPU queries
Run as container root on Ubuntu22.04 x86_64. Existing dirty source trees are preserved.
HELP
            exit 0 ;;
        --dry-run)
            if command -v python3 >/dev/null 2>&1; then
                exec python3 -B "$setup_dir/setup_h100.py" "$@"
            fi
            printf '%s\n' 'PLAN ONLY: bootstrap python3/curl/git/bzip2; Miniforge; Python3.10.20;' \
                'pinned Megatron/submodules; CUDA12.4 + existing12.8; PyTorch2.6.0+cu124; Apex; validation.'
            exit 0 ;;
    esac
done

[[ $EUID == 0 ]] || { echo 'Run this script as root INSIDE the training container.' >&2; exit 1; }
source /etc/os-release
[[ ${ID:-} == ubuntu && ${VERSION_ID:-} == 22.04 && $(uname -m) == x86_64 ]] || {
    echo 'This setup targets the Ubuntu22.04 x86_64 CUDA container.' >&2; exit 1;
}
unset PYTHONPATH PYTHONHOME PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
# The CUDA development image need not contain Python or Git. Bootstrap only
# these OS utilities; the Python controller checks all CUDA/compiler installs.
setup_missing=()
# ZoneInfo('Asia/Shanghai') is used before Conda exists. Minimal CUDA images
# need the OS timezone database as well as the bootstrap Python interpreter.
for setup_package in python3 curl git bzip2 ca-certificates tzdata; do
    if [[ $(dpkg-query -W -f='${Status}' "$setup_package" 2>/dev/null || true) != 'install ok installed' ]]; then
        setup_missing+=("$setup_package")
    fi
done
if (( ${#setup_missing[@]} )); then
    export DEBIAN_FRONTEND=noninteractive
    export TZ=Asia/Shanghai
    apt-get -o Dpkg::Lock::Timeout=120 -o Acquire::Retries=3 update
    apt-get -o Dpkg::Lock::Timeout=120 -o Acquire::Retries=3 install -y --no-install-recommends "${setup_missing[@]}"
fi
exec python3 -B "$setup_dir/setup_h100.py" "$@"
