#!/usr/bin/env bash
set -euo pipefail

# Download and extract MME benchmark data under data/MME by default.
REPO_ID="darkyarding/MME"
MME_ROOT="${MME_ROOT:-data/MME}"
ZIP_NAME="MME_Benchmark_release_version.zip"
ZIP_PATH="${MME_ROOT}/${ZIP_NAME}"
EXTRACT_DIR="${MME_ROOT}/MME_Benchmark_release_version"
export REPO_ID MME_ROOT ZIP_NAME

mkdir -p "${MME_ROOT}"

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "[info] downloading ${ZIP_NAME} from ${REPO_ID} ..."
  python - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id = os.environ.get("REPO_ID", "darkyarding/MME")
local_dir = os.environ["MME_ROOT"]
filename = os.environ.get("ZIP_NAME", "MME_Benchmark_release_version.zip")

path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=filename,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"downloaded: {path}")
PY
else
  echo "[info] zip already exists: ${ZIP_PATH}"
fi

if [[ ! -d "${EXTRACT_DIR}" ]]; then
  echo "[info] extracting ${ZIP_PATH} ..."
  unzip -q -o "${ZIP_PATH}" -d "${MME_ROOT}"
else
  echo "[info] extracted dir already exists: ${EXTRACT_DIR}"
fi

echo "[done] MME data ready: ${EXTRACT_DIR}"
