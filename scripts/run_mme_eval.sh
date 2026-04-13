#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/opsd/output/opsd_full_4gpu/opsd_vcd_single_teacher_e1_cap1000_nccl_20260414_020000}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
PROCESSOR_PATH=${PROCESSOR_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}

MME_ROOT=${MME_ROOT:-data/MME}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/eval_mme_$(date +%Y%m%d_%H%M%S)}
BATCH_SIZE=${BATCH_SIZE:-8}
TORCH_DTYPE=${TORCH_DTYPE:-float16}
PARALLEL_GPUS=${PARALLEL_GPUS:-4}

echo "[config] MODEL_PATH=${MODEL_PATH}"
if [[ "${BASE_MODEL_PATH,,}" == "none" ]]; then
  BASE_MODEL_PATH=""
fi
if [[ -n "${BASE_MODEL_PATH}" ]]; then
  BASE_MODEL_ARG=(--base-model-path "${BASE_MODEL_PATH}")
  echo "[config] BASE_MODEL_PATH=${BASE_MODEL_PATH}"
else
  BASE_MODEL_ARG=()
  echo "[config] BASE_MODEL_PATH=<none>"
fi
echo "[config] PROCESSOR_PATH=${PROCESSOR_PATH}"
echo "[config] MME_ROOT=${MME_ROOT}"
echo "[config] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[config] BATCH_SIZE=${BATCH_SIZE}"
echo "[config] PARALLEL_GPUS=${PARALLEL_GPUS}"

bash scripts/prepare_mme_data.sh

ANSWERS_FILE="${OUTPUT_ROOT}/mme_answers.jsonl"
METRICS_FILE="${OUTPUT_ROOT}/mme_metrics.json"
SUMMARY_FILE="${OUTPUT_ROOT}/mme_summary.md"

mkdir -p "${OUTPUT_ROOT}"

if [[ "${PARALLEL_GPUS}" -gt 1 ]]; then
  CHUNK_DIR="${OUTPUT_ROOT}/chunks"
  mkdir -p "${CHUNK_DIR}"

  pids=()
  for ((i=0; i<PARALLEL_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="${i}" python eval/evaluate_mme_qwen25vl.py \
      --model-path "${MODEL_PATH}" \
      "${BASE_MODEL_ARG[@]}" \
      --processor-path "${PROCESSOR_PATH}" \
      --mme-root "${MME_ROOT}" \
      --answers-file "${CHUNK_DIR}/answers_chunk_${i}.jsonl" \
      --metrics-file "${CHUNK_DIR}/metrics_chunk_${i}.json" \
      --summary-file "${CHUNK_DIR}/summary_chunk_${i}.md" \
      --batch-size "${BATCH_SIZE}" \
      --torch-dtype "${TORCH_DTYPE}" \
      --num-chunks "${PARALLEL_GPUS}" \
      --chunk-idx "${i}" \
      > "${CHUNK_DIR}/chunk_${i}.log" 2>&1 &
    pid=$!
    pids+=("${pid}")
    echo "[launch] chunk ${i} pid=${pid} gpu=${i}"
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[error] one or more chunk jobs failed"
    exit 1
  fi

  cat "${CHUNK_DIR}"/answers_chunk_*.jsonl > "${ANSWERS_FILE}"

  python - <<PY
import json
from pathlib import Path
from eval.evaluate_mme_qwen25vl import compute_mme_metrics, write_summary_markdown

answers_file = Path("${ANSWERS_FILE}")
metrics_file = Path("${METRICS_FILE}")
summary_file = Path("${SUMMARY_FILE}")

records = []
with answers_file.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

metrics = compute_mme_metrics(records)
metrics_file.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
write_summary_markdown(summary_file, metrics)
print(f"[score] total={metrics['totals']['total']:.2f}, perception={metrics['totals']['perception']:.2f}, cognition={metrics['totals']['cognition']:.2f}")
PY
else
  python eval/evaluate_mme_qwen25vl.py \
    --model-path "${MODEL_PATH}" \
    "${BASE_MODEL_ARG[@]}" \
    --processor-path "${PROCESSOR_PATH}" \
    --mme-root "${MME_ROOT}" \
    --answers-file "${ANSWERS_FILE}" \
    --metrics-file "${METRICS_FILE}" \
    --summary-file "${SUMMARY_FILE}" \
    --batch-size "${BATCH_SIZE}" \
    --torch-dtype "${TORCH_DTYPE}"
fi

echo "[done] summary: ${SUMMARY_FILE}"
