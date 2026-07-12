#!/usr/bin/env bash
set -euo pipefail

config="${1:-nested.toml}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/microhh-mpl}"
mkdir -p "${MPLCONFIGDIR}"

ndomains="$(uv run nested_input.py --config="${config}" --list-domains)"

./run_domain.sh warmup "${config}"

for ((domain = 0; domain < ndomains; domain++)); do
    ./run_domain.sh run "${domain}" "${config}"
done
