#!/usr/bin/env bash
set -euo pipefail

BASELINE_MODEL_PATH=${BASELINE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
OURS_MODEL_PATH=${OURS_MODEL_PATH:-/root/autodl-tmp/opsd/output/opsd_full_4gpu/opsd_vcd_single_teacher_e1_cap1000_nccl_20260414_020000}
OURS_BASE_MODEL_PATH=${OURS_BASE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
PROCESSOR_PATH=${PROCESSOR_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
MME_ROOT=${MME_ROOT:-data/MME}

OUTPUT_ROOT=${OUTPUT_ROOT:-output/eval_mme_compare_$(date +%Y%m%d_%H%M%S)}
BATCH_SIZE=${BATCH_SIZE:-8}
PARALLEL_GPUS=${PARALLEL_GPUS:-4}
TORCH_DTYPE=${TORCH_DTYPE:-float16}

BASELINE_OUTPUT_ROOT="${OUTPUT_ROOT}/baseline"
OURS_OUTPUT_ROOT="${OUTPUT_ROOT}/ours"
COMPARE_MD="${OUTPUT_ROOT}/mme_compare_ours_vs_baseline.md"

echo "[compare] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[compare] BASELINE_MODEL_PATH=${BASELINE_MODEL_PATH}"
echo "[compare] OURS_MODEL_PATH=${OURS_MODEL_PATH}"
echo "[compare] OURS_BASE_MODEL_PATH=${OURS_BASE_MODEL_PATH}"
echo "[compare] MME_ROOT=${MME_ROOT}"
echo "[compare] BATCH_SIZE=${BATCH_SIZE}, PARALLEL_GPUS=${PARALLEL_GPUS}"

mkdir -p "${OUTPUT_ROOT}"

echo "[step] run baseline with the same eval pipeline"
OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT}" \
MME_ROOT="${MME_ROOT}" \
MODEL_PATH="${BASELINE_MODEL_PATH}" \
BASE_MODEL_PATH="none" \
PROCESSOR_PATH="${PROCESSOR_PATH}" \
BATCH_SIZE="${BATCH_SIZE}" \
PARALLEL_GPUS="${PARALLEL_GPUS}" \
TORCH_DTYPE="${TORCH_DTYPE}" \
bash scripts/run_mme_eval.sh

echo "[step] run ours with the same eval pipeline"
OUTPUT_ROOT="${OURS_OUTPUT_ROOT}" \
MME_ROOT="${MME_ROOT}" \
MODEL_PATH="${OURS_MODEL_PATH}" \
BASE_MODEL_PATH="${OURS_BASE_MODEL_PATH}" \
PROCESSOR_PATH="${PROCESSOR_PATH}" \
BATCH_SIZE="${BATCH_SIZE}" \
PARALLEL_GPUS="${PARALLEL_GPUS}" \
TORCH_DTYPE="${TORCH_DTYPE}" \
bash scripts/run_mme_eval.sh

echo "[step] build compare markdown"
python - <<PY
import json
from pathlib import Path

ours_path = Path("${OURS_OUTPUT_ROOT}/mme_metrics.json")
base_path = Path("${BASELINE_OUTPUT_ROOT}/mme_metrics.json")
out_path = Path("${COMPARE_MD}")

ours = json.loads(ours_path.read_text(encoding="utf-8"))
base = json.loads(base_path.read_text(encoding="utf-8"))

cats = sorted(set(ours["per_category"]).union(base["per_category"]))

def g(d, cat, key):
    return float(d.get("per_category", {}).get(cat, {}).get(key, 0.0))

lines = []
lines.append("# MME Ours vs Baseline")
lines.append("")
lines.append("| Metric | Ours | Baseline | Delta (Ours-Baseline) |")
lines.append("|---|---:|---:|---:|")
for k, label in [("total", "Total"), ("perception", "Perception"), ("cognition", "Cognition")]:
    o = float(ours["totals"][k])
    b = float(base["totals"][k])
    lines.append(f"| {label} | {o:.2f} | {b:.2f} | {o-b:+.2f} |")

lines.append("")
lines.append("| Category | Ours Score | Baseline Score | Delta | Ours Acc | Baseline Acc | Acc Delta | Ours Acc+ | Baseline Acc+ | Acc+ Delta |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for c in cats:
    os = g(ours, c, "score")
    bs = g(base, c, "score")
    oa = g(ours, c, "acc")
    ba = g(base, c, "acc")
    op = g(ours, c, "acc_plus")
    bp = g(base, c, "acc_plus")
    lines.append(
        f"| {c} | {os:.2f} | {bs:.2f} | {os-bs:+.2f} | {oa:.2f} | {ba:.2f} | {oa-ba:+.2f} | {op:.2f} | {bp:.2f} | {op-bp:+.2f} |"
    )

out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[done] compare file: {out_path}")
print(f"[score] total_delta={float(ours['totals']['total']) - float(base['totals']['total']):+.2f}")
PY

echo "[done] baseline summary: ${BASELINE_OUTPUT_ROOT}/mme_summary.md"
echo "[done] ours summary: ${OURS_OUTPUT_ROOT}/mme_summary.md"
echo "[done] compare markdown: ${COMPARE_MD}"
