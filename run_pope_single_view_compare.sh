#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${MODEL_PATH}}"
POPE_ROOT="${POPE_ROOT:-${REPO_ROOT}/data/POPE/coco}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${REPO_ROOT}/data/coco/val2014}"
SPLITS="${SPLITS:-random popular adversarial}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/eval_pope_single_view_compare_$(date +%Y%m%d_%H%M%S)}"
ANSWER_DIR="${OUTPUT_ROOT}/answers"
METRIC_DIR="${OUTPUT_ROOT}/metrics"
SUMMARY_MD="${OUTPUT_ROOT}/pope_single_view_summary.md"

TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
SEED="${SEED:-42}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
DEFAULT_GPU_COUNT="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
if [[ -z "${DEFAULT_GPU_COUNT}" || "${DEFAULT_GPU_COUNT}" -le 0 ]]; then
  DEFAULT_GPU_COUNT=1
fi
GEN_PARALLEL_GPUS="${GEN_PARALLEL_GPUS:-${DEFAULT_GPU_COUNT}}"

NOISE_STD="${NOISE_STD:-25.0}"
NOISE_STEPS="${NOISE_STEPS:-500}"
MASK_RATIO="${MASK_RATIO:-0.25}"
MASK_MIN_RATIO="${MASK_MIN_RATIO:-$MASK_RATIO}"
MASK_MAX_RATIO="${MASK_MAX_RATIO:-$MASK_RATIO}"
MASK_COUNT="${MASK_COUNT:-1}"
BLUR_RADIUS="${BLUR_RADIUS:-2.0}"
FG_MASK_KEEP_RATIO="${FG_MASK_KEEP_RATIO:-0.35}"
FG_MASK_CENTER_BIAS="${FG_MASK_CENTER_BIAS:-0.15}"

mkdir -p "${ANSWER_DIR}" "${METRIC_DIR}"

run_one_view() {
  local tag="$1"
  for split in ${SPLITS}; do
    local question_file="${POPE_ROOT}/coco_pope_${split}.json"
    local answer_file="${ANSWER_DIR}/${tag}_coco_pope_${split}.jsonl"
    local metric_file="${METRIC_DIR}/${tag}_coco_pope_${split}.json"

    echo "[run] view=${tag}, split=${split}"

    if [[ "${GEN_PARALLEL_GPUS}" -gt 1 ]]; then
      local chunk_dir="${ANSWER_DIR}/.chunks_${tag}_${split}"
      rm -rf "${chunk_dir}"
      mkdir -p "${chunk_dir}"

      local pids=()
      for ((chunk_idx=0; chunk_idx<GEN_PARALLEL_GPUS; chunk_idx++)); do
        local chunk_file="${chunk_dir}/chunk_${chunk_idx}.jsonl"
        cmd=(
          /root/miniconda3/envs/opsd/bin/python "${REPO_ROOT}/eval/object_hallucination_vqa_qwen25vl.py"
          --model-path "${MODEL_PATH}"
          --processor-path "${PROCESSOR_PATH}"
          --image-folder "${IMAGE_FOLDER}"
          --question-file "${question_file}"
          --answers-file "${chunk_file}"
          --input-mode multimodal
          --image-perturbation "${tag}"
          --perturb-noise-std "${NOISE_STD}"
          --perturb-noise-steps "${NOISE_STEPS}"
          --perturb-mask-ratio "${MASK_RATIO}"
          --perturb-mask-min-ratio "${MASK_MIN_RATIO}"
          --perturb-mask-max-ratio "${MASK_MAX_RATIO}"
          --perturb-mask-count "${MASK_COUNT}"
          --perturb-blur-radius "${BLUR_RADIUS}"
          --perturb-fg-mask-keep-ratio "${FG_MASK_KEEP_RATIO}"
          --perturb-fg-mask-center-bias "${FG_MASK_CENTER_BIAS}"
          --batch-size "${GEN_BATCH_SIZE}"
          --temperature "${TEMPERATURE}"
          --top_p "${TOP_P}"
          --max-new-tokens "${MAX_NEW_TOKENS}"
          --seed "${SEED}"
          --torch-dtype "${TORCH_DTYPE}"
          --attn-implementation "${ATTN_IMPLEMENTATION}"
          --num-chunks "${GEN_PARALLEL_GPUS}"
          --chunk-idx "${chunk_idx}"
        )

        if [[ -n "${TOP_K}" ]]; then
          cmd+=(--top_k "${TOP_K}")
        fi

        CUDA_VISIBLE_DEVICES="${chunk_idx}" "${cmd[@]}" &
        pids+=("$!")
      done

      local any_failed=0
      for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
          any_failed=1
        fi
      done
      if [[ "${any_failed}" -ne 0 ]]; then
        echo "[error] One or more chunk jobs failed for view=${tag}, split=${split}" >&2
        exit 1
      fi

      : > "${answer_file}"
      for ((chunk_idx=0; chunk_idx<GEN_PARALLEL_GPUS; chunk_idx++)); do
        cat "${chunk_dir}/chunk_${chunk_idx}.jsonl" >> "${answer_file}"
      done
      rm -rf "${chunk_dir}"
    else
      cmd=(
        /root/miniconda3/envs/opsd/bin/python "${REPO_ROOT}/eval/object_hallucination_vqa_qwen25vl.py"
        --model-path "${MODEL_PATH}"
        --processor-path "${PROCESSOR_PATH}"
        --image-folder "${IMAGE_FOLDER}"
        --question-file "${question_file}"
        --answers-file "${answer_file}"
        --input-mode multimodal
        --image-perturbation "${tag}"
        --perturb-noise-std "${NOISE_STD}"
        --perturb-noise-steps "${NOISE_STEPS}"
        --perturb-mask-ratio "${MASK_RATIO}"
        --perturb-mask-min-ratio "${MASK_MIN_RATIO}"
        --perturb-mask-max-ratio "${MASK_MAX_RATIO}"
        --perturb-mask-count "${MASK_COUNT}"
        --perturb-blur-radius "${BLUR_RADIUS}"
        --perturb-fg-mask-keep-ratio "${FG_MASK_KEEP_RATIO}"
        --perturb-fg-mask-center-bias "${FG_MASK_CENTER_BIAS}"
        --batch-size "${GEN_BATCH_SIZE}"
        --temperature "${TEMPERATURE}"
        --top_p "${TOP_P}"
        --max-new-tokens "${MAX_NEW_TOKENS}"
        --seed "${SEED}"
        --torch-dtype "${TORCH_DTYPE}"
        --attn-implementation "${ATTN_IMPLEMENTATION}"
      )

      if [[ -n "${TOP_K}" ]]; then
        cmd+=(--top_k "${TOP_K}")
      fi

      "${cmd[@]}"
    fi

    /root/miniconda3/envs/opsd/bin/python "${REPO_ROOT}/eval/eval_pope.py" \
      --gt_files "${question_file}" \
      --gen_files "${answer_file}" \
      --strict_order \
      --out_file "${metric_file}"
  done
}

run_one_view clean
run_one_view noise

/root/miniconda3/envs/opsd/bin/python "${REPO_ROOT}/eval/summarize_pope_metrics.py" \
  --metrics-dir "${METRIC_DIR}" \
  --output-file "${SUMMARY_MD}"

echo "[done] summary=${SUMMARY_MD}"
