#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./run_domain.sh warmup [CONFIG]
  ./run_domain.sh run DOMAIN [CONFIG]

Examples:
  ./run_domain.sh warmup nested.toml
  ./run_domain.sh run 0 nested.toml
  ./run_domain.sh run 1 nested.toml
EOF
}

run_name=""
if [[ "${1:-}" == "warmup" ]]; then
    run_name="warmup"
    config="${2:-nested.toml}"
    domain="0"
elif [[ "${1:-}" == "run" ]]; then
    domain="${2:?missing domain index}"
    config="${3:-nested.toml}"
    run_name="run"
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
else
    usage >&2
    exit 2
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/microhh-mpl}"
mkdir -p "${MPLCONFIGDIR}"

mapfile -t run_info < <(
uv run python - "$config" "${domain}" "${run_name}" <<'PY'
import sys
import tomllib
from pathlib import Path

cfg = tomllib.loads(Path(sys.argv[1]).read_text())
domain_arg = int(sys.argv[2])
run_name = sys.argv[3]
run = None

if run_name == "warmup":
    run = {"init": True, "clean": True, "run": True}
    i = 0
    run_dir = "warmup"
elif run_name == "run":
    if "run" not in cfg:
        raise SystemExit("Run command requires [run] in the TOML.")
    i = domain_arg
    run = {"init": i > 0, "clean": True, "run": True}
    run_dir = cfg["domains"][i].get("name", f"dom{i}")
else:
    raise SystemExit(f"Unknown run command: {run_name}")

print(i)
print(run_dir)
print(cfg["case"].get("microhh_name", "era5_openbc"))
print(str(run.get("init", True) if run else True).lower())
print(str(run.get("clean", True) if run else True).lower())
print(str(run.get("run", True) if run else True).lower())
PY
)

domain="${run_info[0]}"
dom_name="${run_info[1]}"
case_name="${run_info[2]}"
do_init="${run_info[3]}"
do_clean="${run_info[4]}"
do_run="${run_info[5]}"

mkdir -p "${dom_name}"
if [[ "${do_clean}" == "true" ]]; then
    find "${dom_name}" -mindepth 1 -maxdepth 1 ! -name microhh -exec rm -rf {} +
fi
ln -sf "${MICROHH_BIN:-../../../build/microhh}" "${dom_name}/microhh"

generator_args=(nested_input.py --config="${config}" --domain="${domain}")
if [[ -n "${run_name}" ]]; then
    generator_args+=(--run-name "${run_name}")
fi
uv run "${generator_args[@]}"

cd "${dom_name}"

if [[ "${do_init}" == "true" ]]; then
    ./microhh init "${case_name}"

    find . -maxdepth 1 -type f -name '*_overwrite*' | while read -r file; do
        newname="${file/_overwrite/}"
        echo "Renaming: ${file} -> ${newname}"
        mv "${file}" "${newname}"
    done
fi

if [[ "${do_run}" == "true" ]]; then
    ./microhh run "${case_name}"
fi
