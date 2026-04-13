#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

POPE_DIR="${POPE_DIR:-${REPO_ROOT}/data/POPE/coco}"
COCO_LINK_DIR="${COCO_LINK_DIR:-${REPO_ROOT}/data/coco/val2014}"
COCO_VAL2014_SRC="${COCO_VAL2014_SRC:-}"

mkdir -p "${POPE_DIR}"

download_one() {
  local name="$1"
  local dst="${POPE_DIR}/coco_pope_${name}.json"
  local urls=(
    "https://raw.githubusercontent.com/DAMO-NLP-SG/VCD/master/experiments/data/POPE/coco/coco_pope_${name}.json"
    "https://cdn.jsdelivr.net/gh/DAMO-NLP-SG/VCD@master/experiments/data/POPE/coco/coco_pope_${name}.json"
  )

  if [[ -f "${dst}" ]]; then
    echo "[skip] ${dst} already exists"
    return
  fi

  for url in "${urls[@]}"; do
    echo "[download] ${url}"
    if curl --http1.1 -L --fail --retry 1 --retry-delay 1 --connect-timeout 15 --max-time 90 -o "${dst}" "${url}"; then
      return
    fi
    echo "[warn] download failed from ${url}"
  done

  echo "[error] failed to download coco_pope_${name}.json from all sources" >&2
  exit 1
}

download_one random
download_one popular
download_one adversarial

echo "[ok] POPE annotations prepared at ${POPE_DIR}"

if [[ -n "${COCO_VAL2014_SRC}" ]]; then
  if [[ ! -d "${COCO_VAL2014_SRC}" ]]; then
    echo "[error] COCO_VAL2014_SRC does not exist: ${COCO_VAL2014_SRC}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${COCO_LINK_DIR}")"
  rm -rf "${COCO_LINK_DIR}"
  ln -s "${COCO_VAL2014_SRC}" "${COCO_LINK_DIR}"
  echo "[ok] Linked COCO val2014: ${COCO_LINK_DIR} -> ${COCO_VAL2014_SRC}"
else
  echo "[warn] COCO_VAL2014_SRC is empty."
  echo "[warn] Please set COCO_VAL2014_SRC=/path/to/val2014 and rerun this script,"
  echo "[warn] or manually place images under ${COCO_LINK_DIR}."
fi
