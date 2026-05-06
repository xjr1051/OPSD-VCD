#!/usr/bin/env bash
set -euo pipefail

BASELINE_MODEL_PATH=${BASELINE_MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
OURS_MODEL_PATH=${OURS_MODEL_PATH:-/root/autodl-tmp/opsd/output/opsd_full_4gpu/opsd_vcd_single_teacher_e1_cap1000_nccl_20260414_020000}
OURS_BASE_MODEL_PATH=${OURS_BASE_MODEL_PATH-"/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"}
PROCESSOR_PATH=${PROCESSOR_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
MMHAL_ROOT=${MMHAL_ROOT:-data/MMHal-Bench}

OUTPUT_ROOT=${OUTPUT_ROOT:-output/eval_mmhal_compare_$(date +%Y%m%d_%H%M%S)}
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
JUDGE_API_BASE=${JUDGE_API_BASE:-}
JUDGE_API_KEY=${JUDGE_API_KEY:-}
BASELINE_USE_VCD_DECODING=${BASELINE_USE_VCD_DECODING:-0}
BASELINE_VCD_ALPHA=${BASELINE_VCD_ALPHA:-1.0}
BASELINE_VCD_BETA=${BASELINE_VCD_BETA:-0.1}
BASELINE_VCD_GAMMA=${BASELINE_VCD_GAMMA:-0.1}
BASELINE_VCD_NOISE_STEPS=${BASELINE_VCD_NOISE_STEPS:-500}
BASELINE_VCD_VIEW_PAIR=${BASELINE_VCD_VIEW_PAIR:-clean-noise}
BASELINE_VCD_START_STEP=${BASELINE_VCD_START_STEP:-0}
BASELINE_VCD_MAX_STEPS=${BASELINE_VCD_MAX_STEPS:-}
BASELINE_VCD_MIN_KEEP=${BASELINE_VCD_MIN_KEEP:-0}
OURS_USE_VCD_DECODING=${OURS_USE_VCD_DECODING:-0}
OURS_VCD_ALPHA=${OURS_VCD_ALPHA:-1.0}
OURS_VCD_BETA=${OURS_VCD_BETA:-0.1}
OURS_VCD_GAMMA=${OURS_VCD_GAMMA:-0.1}
OURS_VCD_NOISE_STEPS=${OURS_VCD_NOISE_STEPS:-500}
OURS_VCD_VIEW_PAIR=${OURS_VCD_VIEW_PAIR:-clean-noise}
OURS_VCD_START_STEP=${OURS_VCD_START_STEP:-0}
OURS_VCD_MAX_STEPS=${OURS_VCD_MAX_STEPS:-}
OURS_VCD_MIN_KEEP=${OURS_VCD_MIN_KEEP:-0}

BASELINE_OUTPUT_ROOT="${OUTPUT_ROOT}/baseline"
OURS_OUTPUT_ROOT="${OUTPUT_ROOT}/ours"
COMPARE_MD="${OUTPUT_ROOT}/mmhal_compare_ours_vs_baseline.md"

echo "[compare] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[compare] BASELINE_MODEL_PATH=${BASELINE_MODEL_PATH}"
echo "[compare] OURS_MODEL_PATH=${OURS_MODEL_PATH}"
echo "[compare] OURS_BASE_MODEL_PATH=${OURS_BASE_MODEL_PATH}"
echo "[compare] MMHAL_ROOT=${MMHAL_ROOT}"
echo "[compare] BATCH_SIZE=${BATCH_SIZE}, PARALLEL_GPUS=${PARALLEL_GPUS}"
echo "[compare] RUN_JUDGE=${RUN_JUDGE}, JUDGE_MODEL=${JUDGE_MODEL}"
if [[ -n "${JUDGE_API_BASE}" ]]; then
    echo "[compare] JUDGE_API_BASE=${JUDGE_API_BASE}"
else
    echo "[compare] JUDGE_API_BASE=<openai default>"
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[step] run baseline with the same eval pipeline"
OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT}" \
MMHAL_ROOT="${MMHAL_ROOT}" \
MODEL_PATH="${BASELINE_MODEL_PATH}" \
BASE_MODEL_PATH="none" \
PROCESSOR_PATH="${PROCESSOR_PATH}" \
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
VCD_START_STEP="${BASELINE_VCD_START_STEP}" \
VCD_MAX_STEPS="${BASELINE_VCD_MAX_STEPS}" \
VCD_MIN_KEEP="${BASELINE_VCD_MIN_KEEP}" \
RUN_JUDGE="${RUN_JUDGE}" \
JUDGE_MODEL="${JUDGE_MODEL}" \
JUDGE_SLEEP_SEC="${JUDGE_SLEEP_SEC}" \
JUDGE_API_BASE="${JUDGE_API_BASE}" \
JUDGE_API_KEY="${JUDGE_API_KEY}" \
bash scripts/run_mmhal_eval.sh

echo "[step] run ours with the same eval pipeline"
OUTPUT_ROOT="${OURS_OUTPUT_ROOT}" \
MMHAL_ROOT="${MMHAL_ROOT}" \
MODEL_PATH="${OURS_MODEL_PATH}" \
BASE_MODEL_PATH="${OURS_BASE_MODEL_PATH}" \
PROCESSOR_PATH="${PROCESSOR_PATH}" \
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
VCD_START_STEP="${OURS_VCD_START_STEP}" \
VCD_MAX_STEPS="${OURS_VCD_MAX_STEPS}" \
VCD_MIN_KEEP="${OURS_VCD_MIN_KEEP}" \
RUN_JUDGE="${RUN_JUDGE}" \
JUDGE_MODEL="${JUDGE_MODEL}" \
JUDGE_SLEEP_SEC="${JUDGE_SLEEP_SEC}" \
JUDGE_API_BASE="${JUDGE_API_BASE}" \
JUDGE_API_KEY="${JUDGE_API_KEY}" \
bash scripts/run_mmhal_eval.sh

echo "[step] build compare markdown"
python - <<PY
import json
from collections import defaultdict
from pathlib import Path

base_root = Path("${BASELINE_OUTPUT_ROOT}")
ours_root = Path("${OURS_OUTPUT_ROOT}")
out_path = Path("${COMPARE_MD}")

base_summary = json.loads((base_root / "mmhal_summary.json").read_text(encoding="utf-8"))
ours_summary = json.loads((ours_root / "mmhal_summary.json").read_text(encoding="utf-8"))
base_rows = json.loads((base_root / "mmhal_responses.json").read_text(encoding="utf-8"))
ours_rows = json.loads((ours_root / "mmhal_responses.json").read_text(encoding="utf-8"))
base_judge_path = base_root / "mmhal_judge_summary.json"
ours_judge_path = ours_root / "mmhal_judge_summary.json"

def calc_extras(rows):
    empty = sum(1 for r in rows if not (r.get("model_answer") or "").strip())
    avg_chars = sum(len((r.get("model_answer") or "")) for r in rows) / max(1, len(rows))
    by_type_words = defaultdict(list)
    for r in rows:
        t = r.get("question_type", "unknown")
        by_type_words[t].append(len((r.get("model_answer") or "").split()))
    by_type_avg_words = {k: sum(v) / max(1, len(v)) for k, v in by_type_words.items()}
    return {
        "empty_answers": empty,
        "avg_response_chars": avg_chars,
        "avg_words_by_type": dict(sorted(by_type_avg_words.items())),
    }

base_extra = calc_extras(base_rows)
ours_extra = calc_extras(ours_rows)

def f(x):
    return float(x)

types = sorted(set(base_summary.get("question_type_counts", {})) | set(ours_summary.get("question_type_counts", {})))

lines = []
lines.append("# MMHal Ours vs Baseline")
lines.append("")
lines.append("Note: This is generation-side comparison only. Official MMHal score requires GPT-4 judge.")
lines.append("")
lines.append("| Metric | Ours | Baseline | Delta (Ours-Baseline) |")
lines.append("|---|---:|---:|---:|")

metrics = [
    ("num_samples", f(ours_summary.get("num_samples", 0)), f(base_summary.get("num_samples", 0))),
    ("avg_response_words", f(ours_summary.get("avg_response_words", 0)), f(base_summary.get("avg_response_words", 0))),
    ("avg_response_chars", f(ours_extra.get("avg_response_chars", 0)), f(base_extra.get("avg_response_chars", 0))),
    ("empty_answers", f(ours_extra.get("empty_answers", 0)), f(base_extra.get("empty_answers", 0))),
]

for name, o, b in metrics:
    lines.append(f"| {name} | {o:.4f} | {b:.4f} | {o-b:+.4f} |")

lines.append("")
lines.append("| Question Type | Ours Count | Baseline Count | Count Delta | Ours Avg Words | Baseline Avg Words | Word Delta |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")
for t in types:
    oc = float(ours_summary.get("question_type_counts", {}).get(t, 0))
    bc = float(base_summary.get("question_type_counts", {}).get(t, 0))
    ow = float(ours_extra.get("avg_words_by_type", {}).get(t, 0.0))
    bw = float(base_extra.get("avg_words_by_type", {}).get(t, 0.0))
    lines.append(f"| {t} | {oc:.0f} | {bc:.0f} | {oc-bc:+.0f} | {ow:.4f} | {bw:.4f} | {ow-bw:+.4f} |")

if base_judge_path.exists() and ours_judge_path.exists():
    base_judge = json.loads(base_judge_path.read_text(encoding="utf-8"))
    ours_judge = json.loads(ours_judge_path.read_text(encoding="utf-8"))

    lines.append("")
    lines.append("## Official Judge Metrics")
    lines.append("")
    lines.append("| Metric | Ours | Baseline | Delta (Ours-Baseline) |")
    lines.append("|---|---:|---:|---:|")

    oa = float(ours_judge.get("avg_score", 0.0))
    ba = float(base_judge.get("avg_score", 0.0))
    oh = float(ours_judge.get("hallucination_rate", 0.0))
    bh = float(base_judge.get("hallucination_rate", 0.0))
    lines.append(f"| avg_score | {oa:.4f} | {ba:.4f} | {oa-ba:+.4f} |")
    lines.append(f"| hallucination_rate | {oh:.4f} | {bh:.4f} | {oh-bh:+.4f} |")

    ot = ours_judge.get("avg_score_by_question_type", {})
    bt = base_judge.get("avg_score_by_question_type", {})
    judge_types = sorted(set(ot) | set(bt))
    lines.append("")
    lines.append("| Question Type | Ours Judge Score | Baseline Judge Score | Delta |")
    lines.append("|---|---:|---:|---:|")
    for t in judge_types:
        ovs = float(ot.get(t, 0.0))
        bvs = float(bt.get(t, 0.0))
        lines.append(f"| {t} | {ovs:.4f} | {bvs:.4f} | {ovs-bvs:+.4f} |")

out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[done] compare file: {out_path}")
print(f"[delta] avg_response_words={f(ours_summary.get('avg_response_words', 0)) - f(base_summary.get('avg_response_words', 0)):+.4f}")
print(f"[delta] avg_response_chars={f(ours_extra.get('avg_response_chars', 0)) - f(base_extra.get('avg_response_chars', 0)):+.4f}")
print(f"[delta] empty_answers={f(ours_extra.get('empty_answers', 0)) - f(base_extra.get('empty_answers', 0)):+.0f}")
if base_judge_path.exists() and ours_judge_path.exists():
    base_judge = json.loads(base_judge_path.read_text(encoding='utf-8'))
    ours_judge = json.loads(ours_judge_path.read_text(encoding='utf-8'))
    print(f"[delta] avg_score={float(ours_judge.get('avg_score', 0.0)) - float(base_judge.get('avg_score', 0.0)):+.4f}")
    print(f"[delta] hallucination_rate={float(ours_judge.get('hallucination_rate', 0.0)) - float(base_judge.get('hallucination_rate', 0.0)):+.4f}")
PY

echo "[done] baseline summary: ${BASELINE_OUTPUT_ROOT}/mmhal_summary.json"
echo "[done] ours summary: ${OURS_OUTPUT_ROOT}/mmhal_summary.json"
echo "[done] compare markdown: ${COMPARE_MD}"
if [[ "${OURS_BASE_MODEL_PATH,,}" == "none" ]]; then
  OURS_BASE_MODEL_PATH=""
fi
