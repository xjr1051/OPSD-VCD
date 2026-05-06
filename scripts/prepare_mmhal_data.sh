#!/usr/bin/env bash
set -euo pipefail

REPO_ID=${REPO_ID:-Shengcao1006/MMHal-Bench}
MMHAL_ROOT=${MMHAL_ROOT:-data/MMHal-Bench}

mkdir -p "${MMHAL_ROOT}"

if [[ -f "${MMHAL_ROOT}/response_template.json" && -d "${MMHAL_ROOT}/images" ]]; then
  img_count=$(find "${MMHAL_ROOT}/images" -maxdepth 1 -type f | wc -l | tr -d ' ')
  if [[ "${img_count}" -ge 96 ]]; then
    echo "[skip] MMHal data already ready at ${MMHAL_ROOT} (images=${img_count})"
    echo "[done] MMHal data ready at ${MMHAL_ROOT}"
    exit 0
  fi
fi

python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ.get("REPO_ID", "Shengcao1006/MMHal-Bench")
local_dir = os.environ.get("MMHAL_ROOT", "data/MMHal-Bench")

# In some environments, Xet/CAS backend may intermittently fail with 401.
# Disable it so huggingface_hub falls back to regular HTTPS downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=["response_template.json", "images/*", "eval_gpt4.py", "README.md"],
)

print(f"[done] MMHal snapshot ready: {local_dir}")
PY

if [[ ! -f "${MMHAL_ROOT}/response_template.json" ]]; then
  echo "[error] missing response_template.json under ${MMHAL_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${MMHAL_ROOT}/images" ]]; then
  echo "[error] missing images dir under ${MMHAL_ROOT}" >&2
  exit 1
fi

echo "[done] MMHal data ready at ${MMHAL_ROOT}"
