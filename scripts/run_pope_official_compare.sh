#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
OURS_MODEL_PATH="${OURS_MODEL_PATH:-${REPO_ROOT}/output/opsd_full_4gpu/opsd_vcd_single_teacher_e1_cap1000_nccl_20260414_020000}"
OURS_BASE_MODEL_PATH="${OURS_BASE_MODEL_PATH:-}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${BASELINE_MODEL_PATH}}"

POPE_ROOT="${POPE_ROOT:-${REPO_ROOT}/data/POPE/coco}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${REPO_ROOT}/data/coco/val2014}"
SPLITS="${SPLITS:-random popular adversarial}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/eval_pope_official}"
ANSWER_DIR="${OUTPUT_ROOT}/answers"
METRIC_DIR="${OUTPUT_ROOT}/metrics"
SUMMARY_MD="${OUTPUT_ROOT}/pope_summary.md"

TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SEED="${SEED:-42}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"

DEFAULT_GPU_COUNT="$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d ' ')"
if [[ -z "${DEFAULT_GPU_COUNT}" || "${DEFAULT_GPU_COUNT}" -le 0 ]]; then
  DEFAULT_GPU_COUNT=1
fi
GEN_PARALLEL_GPUS="${GEN_PARALLEL_GPUS:-${DEFAULT_GPU_COUNT}}"

mkdir -p "${ANSWER_DIR}" "${METRIC_DIR}"

run_one_model() {
  local model_tag="$1"
  local model_path="$2"

  for split in ${SPLITS}; do
    local question_file="${POPE_ROOT}/coco_pope_${split}.json"
    local answer_file="${ANSWER_DIR}/${model_tag}_coco_pope_${split}.jsonl"
    local metric_file="${METRIC_DIR}/${model_tag}_coco_pope_${split}.json"

    if [[ ! -f "${question_file}" ]]; then
      echo "[error] Missing POPE question file: ${question_file}" >&2
      exit 1
    fi

    echo "[run] model=${model_tag}, split=${split}"
    if [[ "${GEN_PARALLEL_GPUS}" -gt 1 ]]; then
      local chunk_dir="${ANSWER_DIR}/.chunks_${model_tag}_${split}"
      rm -rf "${chunk_dir}"
      mkdir -p "${chunk_dir}"

      local pids=()
      for ((chunk_idx=0; chunk_idx<GEN_PARALLEL_GPUS; chunk_idx++)); do
        local chunk_file="${chunk_dir}/chunk_${chunk_idx}.jsonl"
        cmd=(
          python "${REPO_ROOT}/eval/object_hallucination_vqa_qwenvl.py"
          --model-path "${model_path}"
          --processor-path "${PROCESSOR_PATH}"
          --image-folder "${IMAGE_FOLDER}"
          --question-file "${question_file}"
          --answers-file "${chunk_file}"
          --num-chunks "${GEN_PARALLEL_GPUS}"
          --chunk-idx "${chunk_idx}"
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

        if [[ "${model_tag}" == "ours" && -n "${OURS_BASE_MODEL_PATH}" ]]; then
          cmd+=(--base-model-path "${OURS_BASE_MODEL_PATH}")
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
        echo "[error] One or more chunk jobs failed for model=${model_tag}, split=${split}" >&2
        exit 1
      fi

      : > "${answer_file}"
      for ((chunk_idx=0; chunk_idx<GEN_PARALLEL_GPUS; chunk_idx++)); do
        cat "${chunk_dir}/chunk_${chunk_idx}.jsonl" >> "${answer_file}"
      done
      rm -rf "${chunk_dir}"
    else
      cmd=(
        python "${REPO_ROOT}/eval/object_hallucination_vqa_qwenvl.py"
        --model-path "${model_path}"
        --processor-path "${PROCESSOR_PATH}"
        --image-folder "${IMAGE_FOLDER}"
        --question-file "${question_file}"
        --answers-file "${answer_file}"
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

      if [[ "${model_tag}" == "ours" && -n "${OURS_BASE_MODEL_PATH}" ]]; then
        cmd+=(--base-model-path "${OURS_BASE_MODEL_PATH}")
      fi

      "${cmd[@]}"
    fi

    python "${REPO_ROOT}/eval/eval_pope.py" \
      --gt_files "${question_file}" \
      --gen_files "${answer_file}" \
      --strict_order \
      --out_file "${metric_file}"
  done
}

run_one_model baseline "${BASELINE_MODEL_PATH}"
run_one_model ours "${OURS_MODEL_PATH}"

python "${REPO_ROOT}/eval/summarize_pope_metrics.py" \
  --metrics-dir "${METRIC_DIR}" \
  --output-file "${SUMMARY_MD}"

echo "[done] POPE summary: ${SUMMARY_MD}"
