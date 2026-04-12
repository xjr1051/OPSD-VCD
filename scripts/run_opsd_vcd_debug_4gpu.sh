#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU low-VRAM debug launcher for visual OPSD + VCD.
# Usage:
#   MODEL_NAME_OR_PATH=/path/to/vlm DATASET_NAME=<hf_or_local> bash scripts/run_opsd_vcd_debug_4gpu.sh

# IMPORTANT:
# - Use a multimodal model checkpoint (VLM), not a text-only model.
# - This script enables online image perturbation pairs, so dataset must include an image column.

REQUIRED_CONDA_ENV="${REQUIRED_CONDA_ENV:-opsd}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${REQUIRED_CONDA_ENV}" ]]; then
  if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "/root/miniconda3/etc/profile.d/conda.sh"
    conda activate "${REQUIRED_CONDA_ENV}" || {
      echo "[error] Failed to activate conda env: ${REQUIRED_CONDA_ENV}" >&2
      exit 1
    }
  else
    echo "[error] conda init script not found: /root/miniconda3/etc/profile.d/conda.sh" >&2
    exit 1
  fi
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "${REQUIRED_CONDA_ENV}" ]]; then
  echo "[error] This script must run in conda env: ${REQUIRED_CONDA_ENV}" >&2
  exit 1
fi

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
DATASET_NAME="${DATASET_NAME:-derek-thomas/ScienceQA}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-}"
TRAIN_SPLIT="${TRAIN_SPLIT:-validation}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Default to 4 GPUs. You can override at runtime, e.g. NUM_PROCESSES=1 for single-card quick checks.
NUM_PROCESSES="${NUM_PROCESSES:-4}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
MAX_STEPS="${MAX_STEPS:-300}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-128}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/opsd_debug_${NUM_PROCESSES}gpu}"
RUN_CONFIG="${RUN_CONFIG:-opsd_vcd_debug_${NUM_PROCESSES}gpu}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12959}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-accelerate.yaml}"
WANDB_PROJECT="${WANDB_PROJECT:-OPSD-Debug}"
REPORT_TO="${REPORT_TO:-wandb}"
USE_FIXED_TEACHER="${USE_FIXED_TEACHER:-1}"
USE_VCD_OPSD="${USE_VCD_OPSD:-1}"
USE_IMAGE_PERTURBATION_PAIRS="${USE_IMAGE_PERTURBATION_PAIRS:-1}"
USE_PRIVILEGED_VISUAL_TEACHER="${USE_PRIVILEGED_VISUAL_TEACHER:-1}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-1}"
USE_PEFT="${USE_PEFT:-1}"
DDP_BACKEND="${DDP_BACKEND:-nccl}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-}"
AUTO_FALLBACK_NCCL_TO_GLOO="${AUTO_FALLBACK_NCCL_TO_GLOO:-0}"
NCCL_COMPAT_LIB_DIR="${NCCL_COMPAT_LIB_DIR:-/usr/local/cuda-12.8/compat}"
PREFER_CUDA_COMPAT_LIBCUDA="${PREFER_CUDA_COMPAT_LIBCUDA:-1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
LOGGING_STRATEGY="${LOGGING_STRATEGY:-steps}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
LOGGING_FIRST_STEP="${LOGGING_FIRST_STEP:-true}"
INCLUDE_TOKENS_PER_SECOND="${INCLUDE_TOKENS_PER_SECOND:-true}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-true}"

# Field mapping defaults for ScienceQA-style data.
PROBLEM_FIELD="${PROBLEM_FIELD:-question}"
SOLUTION_FIELD="${SOLUTION_FIELD:-lecture}"
IMAGE_FIELD="${IMAGE_FIELD:-image}"
PRIVILEGED_VISUAL_FIELD="${PRIVILEGED_VISUAL_FIELD:-hint}"
IMAGE_TOKEN="${IMAGE_TOKEN:-<|image_pad|>}"

cmd=(
  accelerate launch
  --config_file "${ACCELERATE_CONFIG_FILE}"
  --num_processes "${NUM_PROCESSES}"
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  opsd_train.py
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --learning_rate 2e-6
  --max_grad_norm 0.1
  --per_device_train_batch_size "${PER_DEVICE_BS}"
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}"
  --output_dir "${OUTPUT_DIR}"
  --run_config "${RUN_CONFIG}"
  --num_train_epochs 1
  --max_steps "${MAX_STEPS}"
  --max_completion_length "${MAX_COMPLETION_LENGTH}"
  --logging_strategy "${LOGGING_STRATEGY}"
  --logging_steps "${LOGGING_STEPS}"
  --logging_first_step "${LOGGING_FIRST_STEP}"
  --include_tokens_per_second "${INCLUDE_TOKENS_PER_SECOND}"
  --save_strategy "${SAVE_STRATEGY}"
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --save_only_model "${SAVE_ONLY_MODEL}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --dataloader_pin_memory "${DATALOADER_PIN_MEMORY}"
  --attn_implementation "${ATTN_IMPLEMENTATION}"
  --torch_dtype "${TORCH_DTYPE}"
  --max_length "${MAX_LENGTH}"
  --beta 0
  --use_multimodal_processor
  --temperature 1.0
  --top_p 0.95
  --top_k 20
  --lmbda 1
  --jsd_token_clip 0.05
  --vcd_alpha 0.5
  --view_pairs clean-noise
  --view_field_prefix problem_
  --pair_sampling_strategy first
  --image_field "${IMAGE_FIELD}"
  --image_token "${IMAGE_TOKEN}"
  --noise_std 15.0
  --mask_ratio 0.15
  --blur_radius 1.5
  --privileged_visual_field "${PRIVILEGED_VISUAL_FIELD}"
  --good_view_field problem_good_view
  --bad_view_field problem_bad_view
  --dataset_name "${DATASET_NAME}"
  --train_split "${TRAIN_SPLIT}"
  --problem_field "${PROBLEM_FIELD}"
  --solution_field "${SOLUTION_FIELD}"
  --wandb_project "${WANDB_PROJECT}"
  --report_to "${REPORT_TO}"
)

if [[ -n "${DATASET_CONFIG_NAME}" ]]; then
  cmd+=(--dataset_config_name "${DATASET_CONFIG_NAME}")
fi

if [[ -n "${DDP_BACKEND}" ]]; then
  if [[ "${DDP_BACKEND}" == "nccl" && "${PREFER_CUDA_COMPAT_LIBCUDA}" == "1" ]]; then
    if [[ -f "${NCCL_COMPAT_LIB_DIR}/libcuda.so.1" ]]; then
      export LD_LIBRARY_PATH="${NCCL_COMPAT_LIB_DIR}:${LD_LIBRARY_PATH:-}"
      echo "[run] NCCL libcuda preference enabled: ${NCCL_COMPAT_LIB_DIR}"
    else
      echo "[warn] NCCL compat libcuda path missing: ${NCCL_COMPAT_LIB_DIR}" >&2
    fi
  fi
  if [[ "${DDP_BACKEND}" == "nccl" && "${AUTO_FALLBACK_NCCL_TO_GLOO}" == "1" ]]; then
    echo "[warn] DDP_BACKEND=nccl is unstable in this environment (NCCL init SIGSEGV observed)."
    echo "[warn] Auto-fallback to DDP_BACKEND=gloo. Set AUTO_FALLBACK_NCCL_TO_GLOO=0 to force NCCL."
    DDP_BACKEND="gloo"
  fi
  cmd+=(--ddp_backend "${DDP_BACKEND}")
fi

if [[ -n "${DDP_FIND_UNUSED_PARAMETERS}" ]]; then
  cmd+=(--ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}")
fi

if [[ "${ENABLE_GRADIENT_CHECKPOINTING}" == "1" ]]; then
  cmd+=(--gradient_checkpointing)
fi

if [[ "${USE_PEFT}" == "1" ]]; then
  cmd+=(
    --use_peft
    --lora_r 8
    --lora_alpha 16
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
  )
fi

if [[ "${USE_FIXED_TEACHER}" == "1" ]]; then
  cmd+=(--fixed_teacher)
fi

if [[ "${USE_VCD_OPSD}" == "1" ]]; then
  cmd+=(--use_vcd_opsd)
fi

if [[ "${USE_IMAGE_PERTURBATION_PAIRS}" == "1" ]]; then
  cmd+=(--use_image_perturbation_pairs)
fi

if [[ "${USE_PRIVILEGED_VISUAL_TEACHER}" == "1" ]]; then
  cmd+=(--use_privileged_visual_teacher)
fi

echo "[run] MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH}"
echo "[run] DATASET_NAME=${DATASET_NAME}"
echo "[run] TRAIN_SPLIT=${TRAIN_SPLIT}"
echo "[run] CONDA_ENV=${CONDA_DEFAULT_ENV:-unknown}"
echo "[run] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[run] NUM_PROCESSES=${NUM_PROCESSES}, PER_DEVICE_BS=${PER_DEVICE_BS}, GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"

"${cmd[@]}"
