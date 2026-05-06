#!/usr/bin/env bash
set -euo pipefail

BASELINE_MODEL_PATH=${BASELINE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
OURS_MODEL_PATH=${OURS_MODEL_PATH:-/root/autodl-tmp/opsd/output/opsd_full_4gpu/opsd_vcd_single_teacher_e1_cap1000_nccl_20260414_020000}
OURS_BASE_MODEL_PATH=${OURS_BASE_MODEL_PATH-"/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"}
PROCESSOR_PATH=${PROCESSOR_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
HALLUSION_ROOT=${HALLUSION_ROOT:-data/HallusionBench}

OUTPUT_ROOT=${OUTPUT_ROOT:-output/eval_hallusionbench_compare_$(date +%Y%m%d_%H%M%S)}
BATCH_SIZE=${BATCH_SIZE:-8}
PARALLEL_GPUS=${PARALLEL_GPUS:-4}
TORCH_DTYPE=${TORCH_DTYPE:-float16}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-20}
BASELINE_USE_VCD_DECODING=${BASELINE_USE_VCD_DECODING:-0}
BASELINE_VCD_ALPHA=${BASELINE_VCD_ALPHA:-1.0}
BASELINE_VCD_BETA=${BASELINE_VCD_BETA:-0.1}
BASELINE_VCD_GAMMA=${BASELINE_VCD_GAMMA:-0.1}
BASELINE_VCD_NOISE_STEPS=${BASELINE_VCD_NOISE_STEPS:-500}
BASELINE_VCD_VIEW_PAIR=${BASELINE_VCD_VIEW_PAIR:-clean-noise}
OURS_USE_VCD_DECODING=${OURS_USE_VCD_DECODING:-0}
OURS_VCD_ALPHA=${OURS_VCD_ALPHA:-1.0}
OURS_VCD_BETA=${OURS_VCD_BETA:-0.1}
OURS_VCD_GAMMA=${OURS_VCD_GAMMA:-0.1}
OURS_VCD_NOISE_STEPS=${OURS_VCD_NOISE_STEPS:-500}
OURS_VCD_VIEW_PAIR=${OURS_VCD_VIEW_PAIR:-clean-noise}

BASELINE_OUTPUT_ROOT="${OUTPUT_ROOT}/baseline"
OURS_OUTPUT_ROOT="${OUTPUT_ROOT}/ours"
COMPARE_MD="${OUTPUT_ROOT}/hallusionbench_compare_ours_vs_baseline.md"

mkdir -p "${OUTPUT_ROOT}"

echo "[step] run baseline"
MODEL_PATH="${BASELINE_MODEL_PATH}" \
BASE_MODEL_PATH="none" \
PROCESSOR_PATH="${PROCESSOR_PATH}" \
HALLUSION_ROOT="${HALLUSION_ROOT}" \
OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT}" \
BATCH_SIZE="${BATCH_SIZE}" \
PARALLEL_GPUS="${PARALLEL_GPUS}" \
TORCH_DTYPE="${TORCH_DTYPE}" \
TEMPERATURE="${TEMPERATURE}" \
TOP_P="${TOP_P}" \
TOP_K="${TOP_K}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
USE_VCD_DECODING="${BASELINE_USE_VCD_DECODING}" \
VCD_ALPHA="${BASELINE_VCD_ALPHA}" \
VCD_BETA="${BASELINE_VCD_BETA}" \
VCD_GAMMA="${BASELINE_VCD_GAMMA}" \
VCD_NOISE_STEPS="${BASELINE_VCD_NOISE_STEPS}" \
VCD_VIEW_PAIR="${BASELINE_VCD_VIEW_PAIR}" \
bash scripts/run_hallusionbench_eval.sh

echo "[step] run ours"
MODEL_PATH="${OURS_MODEL_PATH}" \
BASE_MODEL_PATH="${OURS_BASE_MODEL_PATH}" \
PROCESSOR_PATH="${PROCESSOR_PATH}" \
HALLUSION_ROOT="${HALLUSION_ROOT}" \
OUTPUT_ROOT="${OURS_OUTPUT_ROOT}" \
BATCH_SIZE="${BATCH_SIZE}" \
PARALLEL_GPUS="${PARALLEL_GPUS}" \
TORCH_DTYPE="${TORCH_DTYPE}" \
TEMPERATURE="${TEMPERATURE}" \
TOP_P="${TOP_P}" \
TOP_K="${TOP_K}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
USE_VCD_DECODING="${OURS_USE_VCD_DECODING}" \
VCD_ALPHA="${OURS_VCD_ALPHA}" \
VCD_BETA="${OURS_VCD_BETA}" \
VCD_GAMMA="${OURS_VCD_GAMMA}" \
VCD_NOISE_STEPS="${OURS_VCD_NOISE_STEPS}" \
VCD_VIEW_PAIR="${OURS_VCD_VIEW_PAIR}" \
bash scripts/run_hallusionbench_eval.sh

python - <<PY
import json
from pathlib import Path

base_root = Path("${BASELINE_OUTPUT_ROOT}")
ours_root = Path("${OURS_OUTPUT_ROOT}")
out_path = Path("${COMPARE_MD}")

base = json.loads((base_root / "hallusionbench_summary.json").read_text(encoding="utf-8"))
ours = json.loads((ours_root / "hallusionbench_summary.json").read_text(encoding="utf-8"))

metric_map = {
    "qAcc": "Acc per question pair (qAcc)",
    "fAcc": "Acc per figure (fAcc)",
    "easy": "Acc per easy question",
    "hard": "Acc per hard question",
    "aAcc": "Acc per question (aAcc)",
    "VD": "VD",
    "VS": "VS",
    "overall": "Overall",
}

def extract(summary):
    # Backward-compatible key resolution for different score_hallusionbench versions.
    leaderboard = summary.get("leaderboard_metrics") or summary.get("Leaderboard Metrics")
    qa = summary.get("question_accuracy") or summary.get("Question Accuracy")
    if leaderboard is None or qa is None:
        raise KeyError(
            "Missing leaderboard/question accuracy keys. "
            f"Available top-level keys: {list(summary.keys())}"
        )

    def pick(d, *keys):
        for k in keys:
            if k in d:
                return float(d[k])
        raise KeyError(f"Missing keys {keys} in section with keys={list(d.keys())}")

    return {
        "qAcc": pick(leaderboard, "acc_per_question_pair", "Acc per question pair (qAcc)"),
        "fAcc": pick(leaderboard, "acc_per_figure", "Acc per figure (fAcc)"),
        "easy": pick(leaderboard, "acc_per_easy_question", "Acc per easy question"),
        "hard": pick(leaderboard, "acc_per_hard_question", "Acc per hard question"),
        "aAcc": pick(leaderboard, "acc_per_question", "Acc per question (aAcc)"),
        "VD": pick(qa, "VD"),
        "VS": pick(qa, "VS"),
        "overall": pick(qa, "Overall"),
    }

base_m = extract(base)
ours_m = extract(ours)
order = ["qAcc", "fAcc", "easy", "hard", "aAcc", "VD", "VS", "overall"]

lines = [
    "# HallusionBench Ours vs Baseline",
    "",
    "| Metric | Ours | Baseline | Delta (Ours-Baseline) |",
    "|---|---:|---:|---:|",
]
for key in order:
    o = ours_m[key]
    b = base_m[key]
    lines.append(f"| {metric_map[key]} | {o:.4f} | {b:.4f} | {o-b:+.4f} |")

out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[done] compare file: {out_path}")
PY

echo "[done] HallusionBench compare markdown: ${COMPARE_MD}"
if [[ "${OURS_BASE_MODEL_PATH,,}" == "none" ]]; then
  OURS_BASE_MODEL_PATH=""
fi
