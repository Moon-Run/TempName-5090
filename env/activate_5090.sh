#!/usr/bin/env bash
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    echo 'Use: source env/activate_5090.sh' >&2
    exit 1
fi
rtx5090_env_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
rtx5090_env_generated="$rtx5090_env_repo/outputs/5090/env-setup/env.sh"
if [[ ! -f "$rtx5090_env_generated" ]]; then
    echo 'Run bash env/setup_5090.sh successfully first.' >&2
    unset rtx5090_env_repo rtx5090_env_generated
    return 1
fi
# Reject old Megatron/venv activation files from the previous setup protocol.
rtx5090_env_protocol_ok=0
while IFS= read -r rtx5090_env_line; do
    if [[ $rtx5090_env_line == 'export RTX5090_SETUP_PROTOCOL=rtx5090-existing-environment-cutlass-v2' ]]; then
        rtx5090_env_protocol_ok=1
    fi
done < "$rtx5090_env_generated"
if [[ $rtx5090_env_protocol_ok != 1 ]]; then
    echo 'Configuration predates this setup script; rerun bash env/setup_5090.sh.' >&2
    unset rtx5090_env_repo rtx5090_env_generated rtx5090_env_protocol_ok rtx5090_env_line
    return 1
fi
source "$rtx5090_env_generated"
if [[ ! -x "$RTX5090_PYTHON" || ! -x "$NVCC" ]]; then
    echo 'Configured Python/CUDA path is unavailable here; rerun bash env/setup_5090.sh on this server.' >&2
    unset rtx5090_env_repo rtx5090_env_generated rtx5090_env_protocol_ok rtx5090_env_line
    return 1
fi
export PATH="$(dirname -- "$RTX5090_PYTHON"):$CUDA_HOME/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
unset rtx5090_env_repo rtx5090_env_generated rtx5090_env_protocol_ok rtx5090_env_line
