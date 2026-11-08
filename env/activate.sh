#!/usr/bin/env bash
# Source this file; running it as a child cannot modify your current shell.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
    echo 'Use: source env/activate.sh' >&2
    exit 1
fi
h100_env_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
h100_env_generated="$h100_env_repo/outputs/h100/container-recovery/env.sh"
if [[ ! -f "$h100_env_generated" ]]; then
    echo 'Run bash env/setup_h100.sh successfully first.' >&2
    return 1
fi
source "$h100_env_generated"
unset h100_env_repo h100_env_generated
