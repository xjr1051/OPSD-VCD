#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/output/opsd_haloquest_4gpu/opsd_vcd_haloquest_4gpu_len16384_mc32_img768_s1000}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"
PROCESSOR_PATH="${PROCESSOR_PATH:-${BASE_MODEL_PATH:-${MODEL_PATH}}}"

POPE_ROOT="${POPE_ROOT:-${REPO_ROOT}/data/POPE/coco}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${REPO_ROOT}/data/coco/val2014}"
SPLITS="${SPLITS:-random popular adversarial}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/eval_pope_modal_ablation_$(date +%Y%m%d_%H%M%S)}"
ANSWER_DIR="${OUTPUT_ROOT}/answers"
METRIC_DIR="${OUTPUT_ROOT}/metrics"
SUMMARY_MD="${OUTPUT_ROOT}/pope_summary.md"
COMPARE_MD="${OUTPUT_ROOT}/pope_modal_compare.md"

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

run_one_mode() {
  local mode_tag="$1"
  local input_mode="$2"

  for split in ${SPLITS}; do
    local question_file="${POPE_ROOT}/coco_pope_${split}.json"
    local answer_file="${ANSWER_DIR}/${mode_tag}_coco_pope_${split}.jsonl"
    local metric_file="${METRIC_DIR}/${mode_tag}_coco_pope_${split}.json"

    if [[ ! -f "${question_file}" ]]; then
      echo "[error] Missing POPE question file: ${question_file}" >&2
      exit 1
    fi

    echo "[run] mode=${mode_tag}, split=${split}, input_mode=${input_mode}"
    if [[ "${GEN_PARALLEL_GPUS}" -gt 1 ]]; then
      local chunk_dir="${ANSWER_DIR}/.chunks_${mode_tag}_${split}"
      rm -rf "${chunk_dir}"
      mkdir -p "${chunk_dir}"

      local pids=()
      for ((chunk_idx=0; chunk_idx<GEN_PARALLEL_GPUS; chunk_idx++)); do
        local chunk_file="${chunk_dir}/chunk_${chunk_idx}.jsonl"
        cmd=(
          python "${REPO_ROOT}/eval/object_hallucination_vqa_qwenvl.py"
          --model-path "${MODEL_PATH}"
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
          --input-mode "${input_mode}"
        )

        if [[ -n "${TOP_K}" ]]; then
          cmd+=(--top_k "${TOP_K}")
        fi

        if [[ -n "${BASE_MODEL_PATH}" ]]; then
          cmd+=(--base-model-path "${BASE_MODEL_PATH}")
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
        echo "[error] One or more chunk jobs failed for mode=${mode_tag}, split=${split}" >&2
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
        --model-path "${MODEL_PATH}"
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
        --input-mode "${input_mode}"
      )

      if [[ -n "${TOP_K}" ]]; then
        cmd+=(--top_k "${TOP_K}")
      fi

      if [[ -n "${BASE_MODEL_PATH}" ]]; then
        cmd+=(--base-model-path "${BASE_MODEL_PATH}")
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

run_one_mode multimodal multimodal
run_one_mode textonly text-only

python "${REPO_ROOT}/eval/summarize_pope_metrics.py" \
  --metrics-dir "${METRIC_DIR}" \
  --output-file "${SUMMARY_MD}"

python - "${METRIC_DIR}" "${COMPARE_MD}" <<'PY'
import json
import pathlib
import sys

metrics_dir = pathlib.Path(sys.argv[1])
out_file = pathlib.Path(sys.argv[2])
splits = ["random", "popular", "adversarial"]

def read(mode, split):
    path = metrics_dir / f"{mode}_coco_pope_{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))

rows = []
for split in splits:
    mm = read("multimodal", split)
    tx = read("textonly", split)
    rows.append((
        split,
        mm.get("accuracy", 0.0),
        tx.get("accuracy", 0.0),
        mm.get("accuracy", 0.0) - tx.get("accuracy", 0.0),
        mm.get("f1", 0.0),
        tx.get("f1", 0.0),
        mm.get("f1", 0.0) - tx.get("f1", 0.0),
    ))

avg_mm_acc = sum(r[1] for r in rows) / len(rows)
avg_tx_acc = sum(r[2] for r in rows) / len(rows)
avg_mm_f1 = sum(r[4] for r in rows) / len(rows)
avg_tx_f1 = sum(r[5] for r in rows) / len(rows)

lines = []
lines.append("# POPE Multimodal vs Text-only Ablation")
lines.append("")
lines.append("| split | mm_accuracy | text_accuracy | delta(mm-text) | mm_f1 | text_f1 | delta(mm-text) |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for row in rows:
    lines.append(
        f"| {row[0]} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:+.4f} | {row[4]:.4f} | {row[5]:.4f} | {row[6]:+.4f} |"
    )
lines.append(
    f"| avg | {avg_mm_acc:.4f} | {avg_tx_acc:.4f} | {avg_mm_acc - avg_tx_acc:+.4f} | {avg_mm_f1:.4f} | {avg_tx_f1:.4f} | {avg_mm_f1 - avg_tx_f1:+.4f} |"
)
lines.append("")
lines.append("Interpretation: positive delta means image+text beats text-only.")
out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

echo "[done] POPE summary: ${SUMMARY_MD}"
echo "[done] POPE modal compare: ${COMPARE_MD}"
