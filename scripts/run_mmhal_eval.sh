#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/opsd/output/opsd_full_4gpu/opsd_vcd_single_teacher_e1_cap1000_nccl_20260414_020000}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
PROCESSOR_PATH=${PROCESSOR_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/opsd/bin/python}

MMHAL_ROOT=${MMHAL_ROOT:-data/MMHal-Bench}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/eval_mmhal_$(date +%Y%m%d_%H%M%S)}
BATCH_SIZE=${BATCH_SIZE:-8}
PARALLEL_GPUS=${PARALLEL_GPUS:-4}
TORCH_DTYPE=${TORCH_DTYPE:-float16}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
RUN_JUDGE=${RUN_JUDGE:-0}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-4-0314}
JUDGE_SLEEP_SEC=${JUDGE_SLEEP_SEC:-1.0}
OPENAI_API_KEY=${OPENAI_API_KEY:-}
JUDGE_API_KEY=${JUDGE_API_KEY:-${OPENAI_API_KEY}}
JUDGE_API_BASE=${JUDGE_API_BASE:-}
USE_VCD_DECODING=${USE_VCD_DECODING:-0}
VCD_ALPHA=${VCD_ALPHA:-1.0}
VCD_BETA=${VCD_BETA:-0.1}
VCD_GAMMA=${VCD_GAMMA:-0.1}
VCD_NOISE_STEPS=${VCD_NOISE_STEPS:-500}
VCD_VIEW_PAIR=${VCD_VIEW_PAIR:-clean-noise}
VCD_START_STEP=${VCD_START_STEP:-0}
VCD_MAX_STEPS=${VCD_MAX_STEPS:-}
VCD_MIN_KEEP=${VCD_MIN_KEEP:-0}
VCD_NOISE_STD=${VCD_NOISE_STD:-25.0}
VCD_MASK_RATIO=${VCD_MASK_RATIO:-0.25}
VCD_BLUR_RADIUS=${VCD_BLUR_RADIUS:-2.0}
VCD_FG_MASK_KEEP_RATIO=${VCD_FG_MASK_KEEP_RATIO:-0.35}
VCD_FG_MASK_CENTER_BIAS=${VCD_FG_MASK_CENTER_BIAS:-0.15}

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
echo "[config] MMHAL_ROOT=${MMHAL_ROOT}"
echo "[config] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[config] BATCH_SIZE=${BATCH_SIZE}"
echo "[config] PARALLEL_GPUS=${PARALLEL_GPUS}"
echo "[config] TEMPERATURE=${TEMPERATURE}, TOP_P=${TOP_P}, TOP_K=${TOP_K:-<none>}, MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[config] RUN_JUDGE=${RUN_JUDGE}"
echo "[config] JUDGE_MODEL=${JUDGE_MODEL}"
echo "[config] USE_VCD_DECODING=${USE_VCD_DECODING}"
if [[ -n "${JUDGE_API_BASE}" ]]; then
  echo "[config] JUDGE_API_BASE=${JUDGE_API_BASE}"
else
  echo "[config] JUDGE_API_BASE=<openai default>"
fi

VCD_ARGS=()
if [[ "${USE_VCD_DECODING}" == "1" ]]; then
  VCD_ARGS+=(
    --use-vcd-decoding
    --vcd-alpha "${VCD_ALPHA}"
    --vcd-beta "${VCD_BETA}"
    --vcd-gamma "${VCD_GAMMA}"
    --vcd-noise-steps "${VCD_NOISE_STEPS}"
    --vcd-view-pair "${VCD_VIEW_PAIR}"
    --vcd-start-step "${VCD_START_STEP}"
    --vcd-min-keep "${VCD_MIN_KEEP}"
    --vcd-noise-std "${VCD_NOISE_STD}"
    --vcd-mask-ratio "${VCD_MASK_RATIO}"
    --vcd-blur-radius "${VCD_BLUR_RADIUS}"
    --vcd-fg-mask-keep-ratio "${VCD_FG_MASK_KEEP_RATIO}"
    --vcd-fg-mask-center-bias "${VCD_FG_MASK_CENTER_BIAS}"
  )
  if [[ -n "${VCD_MAX_STEPS}" ]]; then
    VCD_ARGS+=(--vcd-max-steps "${VCD_MAX_STEPS}")
  fi
fi

GEN_ARGS=(
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)
if [[ -n "${TOP_K}" ]]; then
  GEN_ARGS+=(--top_k "${TOP_K}")
fi

if [[ "${RUN_JUDGE}" == "1" && -z "${JUDGE_API_KEY}" ]]; then
  echo "[error] RUN_JUDGE=1 but JUDGE_API_KEY/OPENAI_API_KEY is empty" >&2
  exit 1
fi

MMHAL_ROOT="${MMHAL_ROOT}" bash scripts/prepare_mmhal_data.sh

TEMPLATE_FILE="${MMHAL_ROOT}/response_template.json"
IMAGE_FOLDER="${MMHAL_ROOT}/images"
RESPONSES_FILE="${OUTPUT_ROOT}/mmhal_responses.json"
SUMMARY_FILE="${OUTPUT_ROOT}/mmhal_summary.json"
JUDGE_RAW_FILE="${OUTPUT_ROOT}/mmhal_judge_raw.json"
JUDGE_SCORED_FILE="${OUTPUT_ROOT}/mmhal_judge_scored.json"
JUDGE_SUMMARY_FILE="${OUTPUT_ROOT}/mmhal_judge_summary.json"

mkdir -p "${OUTPUT_ROOT}"

if [[ "${PARALLEL_GPUS}" -gt 1 ]]; then
  CHUNK_DIR="${OUTPUT_ROOT}/chunks"
  mkdir -p "${CHUNK_DIR}"

  pids=()
  for ((i=0; i<PARALLEL_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="${i}" "${PYTHON_BIN}" eval/evaluate_mmhal_qwen25vl.py \
      --model-path "${MODEL_PATH}" \
      "${BASE_MODEL_ARG[@]}" \
      --processor-path "${PROCESSOR_PATH}" \
      --template-file "${TEMPLATE_FILE}" \
      --image-folder "${IMAGE_FOLDER}" \
      --output-file "${CHUNK_DIR}/responses_chunk_${i}.json" \
      --summary-file "${CHUNK_DIR}/summary_chunk_${i}.json" \
      --batch-size "${BATCH_SIZE}" \
      --torch-dtype "${TORCH_DTYPE}" \
      --num-chunks "${PARALLEL_GPUS}" \
      --chunk-idx "${i}" \
      "${GEN_ARGS[@]}" \
      "${VCD_ARGS[@]}" \
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

  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

chunk_dir = Path("${CHUNK_DIR}")
out_file = Path("${RESPONSES_FILE}")
summary_file = Path("${SUMMARY_FILE}")

rows = []
for p in sorted(chunk_dir.glob("responses_chunk_*.json")):
    rows.extend(json.loads(p.read_text(encoding="utf-8")))

rows = sorted(rows, key=lambda x: (x.get("id") is None, x.get("id", 10**9)))
out_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

type_counts = {}
for r in rows:
    t = r.get("question_type", "unknown")
    type_counts[t] = type_counts.get(t, 0) + 1
avg_words = sum(len((r.get("model_answer") or "").split()) for r in rows) / max(1, len(rows))

summary = {
    "num_samples": len(rows),
    "avg_response_words": avg_words,
    "question_type_counts": dict(sorted(type_counts.items())),
    "note": "Official MMHal scoring requires GPT-4 judge (OpenAI API). This file contains generation-side statistics only.",
}
summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[done] responses: {out_file}")
print(f"[done] summary: {summary_file}")
PY
else
  "${PYTHON_BIN}" eval/evaluate_mmhal_qwen25vl.py \
    --model-path "${MODEL_PATH}" \
    "${BASE_MODEL_ARG[@]}" \
    --processor-path "${PROCESSOR_PATH}" \
    --template-file "${TEMPLATE_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-file "${RESPONSES_FILE}" \
    --summary-file "${SUMMARY_FILE}" \
    --batch-size "${BATCH_SIZE}" \
    --torch-dtype "${TORCH_DTYPE}" \
    "${GEN_ARGS[@]}" \
    "${VCD_ARGS[@]}"
fi

if [[ "${RUN_JUDGE}" == "1" ]]; then
  echo "[step] run official-style MMHal judge scoring"
  if [[ -n "${JUDGE_API_KEY}" ]]; then
    export JUDGE_API_KEY="${JUDGE_API_KEY}"
  fi

  JUDGE_BASE_ARGS=()
  if [[ -n "${JUDGE_API_BASE}" ]]; then
    JUDGE_BASE_ARGS+=(--api-base "${JUDGE_API_BASE}")
  fi

  "${PYTHON_BIN}" eval/evaluate_mmhal_judge.py \
    --response-file "${RESPONSES_FILE}" \
    --output-file "${JUDGE_RAW_FILE}" \
    --scored-file "${JUDGE_SCORED_FILE}" \
    --summary-file "${JUDGE_SUMMARY_FILE}" \
    "${JUDGE_BASE_ARGS[@]}" \
    --model "${JUDGE_MODEL}" \
    --sleep-sec "${JUDGE_SLEEP_SEC}"

  echo "[done] MMHal judge summary: ${JUDGE_SUMMARY_FILE}"
fi

echo "[done] MMHal outputs: ${OUTPUT_ROOT}"
