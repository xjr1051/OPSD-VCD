#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${MODEL_PATH:-/root/autodl-tmp/opsd/output/opsd_full_4gpu/opsd_vcd_single_teacher_e1_cap1000_nccl_20260414_020000}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
PROCESSOR_PATH=${PROCESSOR_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}

HALLUSION_ROOT=${HALLUSION_ROOT:-data/HallusionBench}
OUTPUT_ROOT=${OUTPUT_ROOT:-output/eval_hallusionbench_$(date +%Y%m%d_%H%M%S)}
BATCH_SIZE=${BATCH_SIZE:-8}
PARALLEL_GPUS=${PARALLEL_GPUS:-4}
TORCH_DTYPE=${TORCH_DTYPE:-float16}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-20}
USE_VCD_DECODING=${USE_VCD_DECODING:-0}
VCD_ALPHA=${VCD_ALPHA:-1.0}
VCD_BETA=${VCD_BETA:-0.1}
VCD_GAMMA=${VCD_GAMMA:-0.1}
VCD_NOISE_STEPS=${VCD_NOISE_STEPS:-500}
VCD_VIEW_PAIR=${VCD_VIEW_PAIR:-clean-noise}
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
echo "[config] HALLUSION_ROOT=${HALLUSION_ROOT}"
echo "[config] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[config] BATCH_SIZE=${BATCH_SIZE}"
echo "[config] PARALLEL_GPUS=${PARALLEL_GPUS}"
echo "[config] TEMPERATURE=${TEMPERATURE}, TOP_P=${TOP_P}, TOP_K=${TOP_K:-<none>}, MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[config] USE_VCD_DECODING=${USE_VCD_DECODING}"

VCD_ARGS=()
if [[ "${USE_VCD_DECODING}" == "1" ]]; then
  VCD_ARGS+=(
    --use-vcd-decoding
    --vcd-alpha "${VCD_ALPHA}"
    --vcd-beta "${VCD_BETA}"
    --vcd-gamma "${VCD_GAMMA}"
    --vcd-noise-steps "${VCD_NOISE_STEPS}"
    --vcd-view-pair "${VCD_VIEW_PAIR}"
    --vcd-noise-std "${VCD_NOISE_STD}"
    --vcd-mask-ratio "${VCD_MASK_RATIO}"
    --vcd-blur-radius "${VCD_BLUR_RADIUS}"
    --vcd-fg-mask-keep-ratio "${VCD_FG_MASK_KEEP_RATIO}"
    --vcd-fg-mask-center-bias "${VCD_FG_MASK_CENTER_BIAS}"
  )
fi

GEN_ARGS=(
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)
if [[ -n "${TOP_K}" ]]; then
  GEN_ARGS+=(--top_k "${TOP_K}")
fi

HALLUSION_ROOT="${HALLUSION_ROOT}" bash scripts/prepare_hallusionbench_data.sh

DATA_FILE="${HALLUSION_ROOT}/HallusionBench.json"
IMAGE_ROOT="${HALLUSION_ROOT}/data"
RESPONSES_FILE="${OUTPUT_ROOT}/hallusionbench_responses.json"
SUMMARY_JSON="${OUTPUT_ROOT}/hallusionbench_summary.json"
SUMMARY_MD="${OUTPUT_ROOT}/hallusionbench_summary.md"

mkdir -p "${OUTPUT_ROOT}"

if [[ "${PARALLEL_GPUS}" -gt 1 ]]; then
  CHUNK_DIR="${OUTPUT_ROOT}/chunks"
  mkdir -p "${CHUNK_DIR}"

  pids=()
  for ((i=0; i<PARALLEL_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES="${i}" python eval/evaluate_hallusionbench_qwen25vl.py \
      --model-path "${MODEL_PATH}" \
      "${BASE_MODEL_ARG[@]}" \
      --processor-path "${PROCESSOR_PATH}" \
      --data-file "${DATA_FILE}" \
      --image-root "${IMAGE_ROOT}" \
      --output-file "${CHUNK_DIR}/responses_chunk_${i}.json" \
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
    echo "[error] one or more HallusionBench chunk jobs failed"
    exit 1
  fi

  python - <<PY
import json
from pathlib import Path

chunk_dir = Path("${CHUNK_DIR}")
out_file = Path("${RESPONSES_FILE}")
rows = []
for p in sorted(chunk_dir.glob('responses_chunk_*.json')):
    rows.extend(json.loads(p.read_text(encoding='utf-8')))
rows = sorted(rows, key=lambda x: x.get('_orig_index', 10**9))
out_file.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')
print(f"[done] responses: {out_file}")
print(f"[info] num_samples={len(rows)}")
PY
else
  python eval/evaluate_hallusionbench_qwen25vl.py \
    --model-path "${MODEL_PATH}" \
    "${BASE_MODEL_ARG[@]}" \
    --processor-path "${PROCESSOR_PATH}" \
    --data-file "${DATA_FILE}" \
    --image-root "${IMAGE_ROOT}" \
    --output-file "${RESPONSES_FILE}" \
    --batch-size "${BATCH_SIZE}" \
    --torch-dtype "${TORCH_DTYPE}" \
    "${GEN_ARGS[@]}" \
    "${VCD_ARGS[@]}"
fi

python eval/score_hallusionbench.py \
  --input-file "${RESPONSES_FILE}" \
  --output-json "${SUMMARY_JSON}" \
  --output-md "${SUMMARY_MD}"

echo "[done] HallusionBench outputs: ${OUTPUT_ROOT}"
echo "[done] responses: ${RESPONSES_FILE}"
echo "[done] summary: ${SUMMARY_MD}"
