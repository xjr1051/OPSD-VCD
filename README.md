# PRISM-OPSD: Privileged Visual Self-Distillation for Grounded VLMs

本仓库是一个新的实验研究项目：在 OPSD 代码基座上，将 VCD 的“原图-扰动图对比”思想迁移到训练阶段，用于降低 VLM 幻觉。

## 1. 核心想法

VCD 告诉我们：对同一图像构造扰动视图，并与 clean 视图做对比解码，可以减轻 hallucination。

本项目把这一思想改造成训练期范式：

- student 在较弱/扰动视图条件下 on-policy 采样轨迹；
- teacher 在较强/更事实视图条件下给出监督；
- teacher 分布通过“teacher-view vs student-view”对比 logits 构造。

当前在线扰动模式支持：

- clean-noise（teacher=clean, student=noise）
- clean-mask（teacher=mask, student=clean）
- clean-blur（teacher=clean, student=blur）

其中 pair 语义固定为 `teacher-student`。

## 2. 方法定义

我们采用带 privileged visual information 的 VLM-OPSD 目标。定义样本为 $(x, v, z^*, y^*)$：

- $x$：文本问题
- $v$：图像
- $z^*$：teacher-only 的特权视觉证据（object list/boxes/masks/scene graph 等）
- $y^*$：参考答案

student on-policy 采样轨迹：

$$
\hat y \sim p_S(\cdot \mid x, v)
$$

token-level distillation 主项：

$$
\mathcal{L}_{\text{V-OPSD}} =
\mathbb{E}_{(x,v,z^*)}
\mathbb{E}_{\hat y \sim p_S(\cdot\mid x,v)}
\left[
\frac{1}{|\hat y|}\sum_t
D\big(
p_T(\cdot\mid x,v,z^*,\hat y_{<t})
\Vert
p_S(\cdot\mid x,v,\hat y_{<t})
\big)
\right]
$$

当前实现严格遵循 OPSD 主目标：on-policy、token-level distillation。

## 3. 当前代码行为

在 `--use_vcd_opsd` 开启时：

- 轨迹来源：student-side 视图（pair 的第二项）；
- teacher 监督：
  - 默认模式：pair 的第一项与第二项构造 teacher 分布；
  - privileged 模式（推荐）：teacher 直接使用 `z^*` 条件给出分布；
- 支持多对视图组合并按策略采样。

## 4. 关键参数

训练入口在 `opsd_train.py`。

### 4.1 Visual OPSD 相关

- `--use_vcd_opsd`：开启视觉版 OPSD 分支（参数名为兼容保留）。
- `--vcd_alpha`：对比强度 $\alpha$。
- `--view_pairs`：teacher-student 视图对（在线扰动模式下由脚本自动映射，单次实验固定一个 pair）。
- `--view_field_prefix`：视图字段前缀，默认 `problem_`。
- `--pair_sampling_strategy`：pair 采样策略（在线扰动模式下固定为 `first`）。
- `--use_image_perturbation_pairs`：在 collator 内对图像在线扰动并生成视图对。
- `--image_field`：原始图像列名，默认 `image`。
- `--image_token`：多模态提示中的图像占位符，默认 `<image>`。
- `--noise_std` / `--mask_ratio` / `--blur_radius`：在线扰动参数。
- `--use_multimodal_processor`：加载 `AutoProcessor`（图片训练必需）。
- `--use_privileged_visual_teacher`：启用 teacher-only 的 privileged visual evidence 条件。
- `--use_single_visual_teacher`：启用纯视觉单 teacher 分支（teacher 仅用 factual/good 视图，不再构造 teacher_bad 对比分支）。
- `--privileged_visual_field`：数据中的 `z^*` 字段名。

脚本级快捷参数（`scripts/run_opsd_vcd_debug_4gpu.sh`）：

- `PERTURBATION_MODE`：`clean-noise | clean-mask | clean-blur`
- `NOISE_STD` / `MASK_RATIO` / `BLUR_RADIUS`：在线扰动强度

说明：
- 当 `USE_IMAGE_PERTURBATION_PAIRS=1` 时，单次实验固定使用一种扰动方式（一个 pair）。
- `PERTURBATION_MODE=clean-mask` 在实现中会映射为 `mask-clean`，即 teacher=mask、student=clean。

### 4.2 数据源与字段（已改为可配置）

- `--dataset_name`：HF 数据集名或本地路径。
- `--dataset_config_name`：可选 config。
- `--train_split`：训练 split。
- `--problem_field`：基础问题字段。
- `--solution_field`：参考解字段。

### 4.3 兼容回退字段

若 pair 对应字段不存在，会回退到：

- `--good_view_field`（teacher/factual 视图）
- `--bad_view_field`（student/perturbed 视图）

## 5. 数据格式

最推荐的数据列组织方式（在线图片扰动模式）：

- 基础列：
  - `problem`
  - `solution`
- 图像列：
  - `image`
- 特权视觉证据列：
  - `privileged_visual_evidence`

可选的数据列组织方式（离线视图文本模式）：

- 视图列（由前缀 + tag 组成，或 legacy good/bad）：
  - `problem_clean`
  - `problem_noise`
  - `problem_mask`
  - ...

如果 `--view_pairs clean-noise,mask-clean`，代码会查找：

- `problem_clean` + `problem_noise`
- `problem_mask` + `problem_clean`

并按 `--pair_sampling_strategy` 选择当步使用的 pair。

## 6. 快速开始

### 6.1 安装

```bash
conda env create -f environment.yml
conda activate opsd
pip install flash-attn==2.8.3 --no-build-isolation
```

### 6.2 运行示例

```bash
bash scripts/run_opsd_vcd_debug_4gpu.sh
```

当前默认主推配置（纯视觉单 teacher）已包含：

- `--use_vcd_opsd`
- `--use_multimodal_processor`
- `--use_image_perturbation_pairs`
- `--use_single_visual_teacher`
- `PERTURBATION_MODE=clean-noise`

默认同时关闭：

- `--use_privileged_visual_teacher`

## 7. 推荐实验矩阵

建议最小实验集：

- alpha 扫描：$\alpha \in \{0.25, 0.5, 1.0, 2.0\}$
- 扰动方式（单次实验固定一种）：
  - clean-noise
  - clean-mask
  - clean-blur

对照组：

- OPSD baseline（关闭 `--use_vcd_opsd`）
- PRISM-OPSD（纯视觉单 teacher，开启 `--use_vcd_opsd` + `--use_single_visual_teacher`）
- Privileged Visual Teacher（作为额外增强分支，不是默认主推）

## 8. 评估建议

按 2311.16922 官方仓库主口径，优先报告 POPE（random / popular / adversarial）。

本仓库已提供与官方脚本同格式的评测链路：

- `eval/object_hallucination_vqa_qwenvl.py`：生成 POPE 答案（JSONL，官方命名对齐）
- `eval/eval_pope.py`：按官方规则计算 Precision / Recall / F1 / Accuracy / yes 比例
- `eval/summarize_pope_metrics.py`：汇总三分集结果
- `scripts/run_pope_official_compare.sh`：一键 baseline vs ours 对比

### 8.1 数据准备

默认目录约定：

- `data/POPE/coco/coco_pope_random.json`
- `data/POPE/coco/coco_pope_popular.json`
- `data/POPE/coco/coco_pope_adversarial.json`
- `data/coco/val2014/`（对应 COCO 图像目录）

可直接用脚本准备 POPE 标注（官方仓库同源）并挂载 COCO 图像目录：

```bash
bash scripts/prepare_pope_data.sh

# 如果你的 COCO val2014 在其他路径，指定后会自动建立软链接到 data/coco/val2014
COCO_VAL2014_SRC=/path/to/val2014 bash scripts/prepare_pope_data.sh
```

### 8.2 一键对比（推荐）

```bash
BASELINE_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
OURS_MODEL_PATH=/root/autodl-tmp/opsd/output/opsd_debug_4gpu/opsd_vcd_debug_4gpu/checkpoint-30 \
bash scripts/run_pope_official_compare.sh
```

输出目录：

- `output/eval_pope_official/answers/`：各 split 生成答案
- `output/eval_pope_official/metrics/`：各 split 指标 JSON
- `output/eval_pope_official/pope_summary.md`：汇总表

### 8.3 手动执行（对齐官方两段式）

1) 生成答案（对应官方 `object_hallucination_vqa_*.py`）

```bash
python eval/object_hallucination_vqa_qwenvl.py \
  --model-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --processor-path /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  --question-file data/POPE/coco/coco_pope_random.json \
  --image-folder data/coco/val2014 \
  --answers-file output/eval_pope_official/answers/baseline_coco_pope_random.jsonl
```

2) 计算 POPE 指标（对应官方 `eval_pope.py`）

```bash
python eval/eval_pope.py \
  --gt_files data/POPE/coco/coco_pope_random.json \
  --gen_files output/eval_pope_official/answers/baseline_coco_pope_random.jsonl \
  --strict_order
```

补充：MME 也属于论文主报告口径之一。当前仓库先提供 POPE 官方口径闭环，MME 可在同一模型输出基础上追加。 

## 9. 重要说明

- 该实现是“训练期融合骨架”，用于将视图对比偏好蒸馏进参数。
- 当前版本已经支持在 collator 中在线生成 clean/noise/mask/blur 视图对。
- `use_image_perturbation_pairs=True` 时，当前不支持 `use_vllm`，请使用 transformers 生成路径。
- 如使用特定 VLM（如 Qwen-VL/LLaVA），请确保 `image_token` 与模型模板一致。

## 10. 仓库结构（核心）

- `opsd_train.py`：训练入口与参数定义。
- `data_collator.py`：teacher-student 视图对组装。
- `opsd_trainer.py`：on-policy 轨迹 + token-level self-distillation 蒸馏。
- `scripts/run_opsd_vcd_debug_4gpu.sh`：当前默认主推启动脚本（纯视觉单 teacher 默认）。
- `eval/object_hallucination_vqa_qwenvl.py`：POPE 答案生成脚本（Qwen2.5-VL，官方命名对齐）。
- `eval/eval_pope.py`：POPE 官方口径指标计算。
- `eval/summarize_pope_metrics.py`：POPE 汇总表生成。
- `scripts/run_pope_official_compare.sh`：POPE baseline vs ours 一键对比。
- `scripts/prepare_mme_data.sh`：MME 数据下载与解压（默认 `darkyarding/MME`）。
- `eval/evaluate_mme_qwen25vl.py`：MME 评测主脚本（支持 `acc`、`acc+`、score 汇总）。
- `scripts/run_mme_eval.sh`：单模型 MME 一键评测（支持 4 卡 chunk 并行）。
- `scripts/run_mme_compare.sh`：同一评测逻辑连续跑 baseline + ours，并输出差值表。
- `scripts/prepare_mmhal_data.sh`：MMHal 数据准备（支持已存在数据快速跳过）。
- `eval/evaluate_mmhal_qwen25vl.py`：MMHal 生成评测脚本（支持 chunk 并行）。
- `scripts/run_mmhal_eval.sh`：单模型 MMHal 一键评测（支持 4 卡 chunk 并行）。
- `scripts/run_mmhal_compare.sh`：同一评测逻辑连续跑 baseline + ours，并输出差值表。

## 11. 训练分支与开启方式（2026-04-13）

本项目当前可用训练方式可以分为 5 类主分支 + 2 类 teacher 更新策略。

### 11.1 主分支总览

| 分支 | 作用 | 关键开关 |
|---|---|---|
| 普通 OPSD（非视觉对比） | 基线蒸馏路径，teacher 单分支监督 student | `use_vcd_opsd=0` |
| 视觉 OPSD（双 teacher 对比） | good/bad 双 teacher 视图对比，构造对比教师分布 | `use_vcd_opsd=1`, `use_image_perturbation_pairs=1`, `use_single_visual_teacher=0`, `use_privileged_visual_teacher=0` |
| 视觉 OPSD（纯视觉单 teacher） | 只用 good/factual teacher 视图监督 weak student 视图 | `use_vcd_opsd=1`, `use_image_perturbation_pairs=1`, `use_single_visual_teacher=1`, `use_privileged_visual_teacher=0` |
| Privileged Visual Teacher | teacher 额外使用 teacher-only 视觉证据文本 `z*` | `use_privileged_visual_teacher=1`, `use_single_visual_teacher=0` |
| Reason-First Teacher | teacher 先分析参考解，再进入教学提示 | `reason_first=1`，且不能与 `use_vcd_opsd=1` 同时开启 |

说明：
- `use_single_visual_teacher=1` 与 `use_privileged_visual_teacher=1` 互斥。
- `use_image_perturbation_pairs=1` 需要 `use_vcd_opsd=1`。
- `reason_first=1` 与 `use_vcd_opsd=1` 互斥。

### 11.2 每个分支具体什么意思

1. 普通 OPSD（非视觉对比）
- student：基于问题（+图像）做 on-policy 采样。
- teacher：单分支提示中包含参考解与 transition prompt。
- 蒸馏：teacher/student 在同一采样轨迹上做 token-level distillation。

2. 视觉 OPSD（双 teacher 对比）
- student：使用 pair 的 student 侧（较弱/扰动）视图采样轨迹。
- teacher：同时跑 good/factual 与 bad/weak 两个分支。
- 蒸馏目标：
  $$
  z_{teacher}=(1+\alpha)z_{good}-\alpha z_{bad}
  $$
- 用于显式抑制 bad 视图偏好的伪相关 token，降低幻觉。

3. 视觉 OPSD（纯视觉单 teacher）
- student：仍在 weak 视图上采样轨迹。
- teacher：只保留 good/factual 视图单分支，不再构造 `teacher_bad`。
- 蒸馏：直接把 good 视图 teacher 分布蒸馏给 student。
- 这是“只靠视觉视图差异做蒸馏”的更直接实现。

4. Privileged Visual Teacher
- teacher 提示中额外注入 `privileged_visual_field`（teacher-only）。
- student 看不到这部分信息。
- 更像“视觉+外部证据增强 teacher”，不是纯视觉视图对比。

5. Reason-First Teacher
- teacher 先读参考解并生成分析，再执行教学提示。
- 属于教师提示策略分支，不是视觉 pair 分支。

### 11.3 如何开启（可直接运行）

以下都以 `scripts/run_opsd_vcd_debug_4gpu.sh` 为入口。

1. 纯视觉单 teacher（推荐，当前脚本默认）

```bash
USE_VCD_OPSD=1 \
USE_IMAGE_PERTURBATION_PAIRS=1 \
USE_SINGLE_VISUAL_TEACHER=1 \
USE_PRIVILEGED_VISUAL_TEACHER=0 \
PERTURBATION_MODE=clean-noise \
bash scripts/run_opsd_vcd_debug_4gpu.sh
```

2. 视觉双 teacher 对比（VCD-OPSD 原始对比分支）

```bash
USE_VCD_OPSD=1 \
USE_IMAGE_PERTURBATION_PAIRS=1 \
USE_SINGLE_VISUAL_TEACHER=0 \
USE_PRIVILEGED_VISUAL_TEACHER=0 \
VCD_ALPHA=0.5 \
PERTURBATION_MODE=clean-blur \
bash scripts/run_opsd_vcd_debug_4gpu.sh
```

3. 普通 OPSD 基线（关闭视觉对比分支）

```bash
USE_VCD_OPSD=0 \
USE_IMAGE_PERTURBATION_PAIRS=0 \
USE_SINGLE_VISUAL_TEACHER=0 \
USE_PRIVILEGED_VISUAL_TEACHER=0 \
bash scripts/run_opsd_vcd_debug_4gpu.sh
```

4. Privileged Visual Teacher

```bash
USE_VCD_OPSD=1 \
USE_IMAGE_PERTURBATION_PAIRS=1 \
USE_SINGLE_VISUAL_TEACHER=0 \
USE_PRIVILEGED_VISUAL_TEACHER=1 \
PRIVILEGED_VISUAL_FIELD=hint \
bash scripts/run_opsd_vcd_debug_4gpu.sh
```

### 11.4 视图对从哪里来：在线图像扰动 vs 数据集字段

有两种来源路径：

1. 在线图像扰动（推荐）
- 条件：`use_image_perturbation_pairs=1`。
- 输入：数据集只需提供原图 `image_field`。
- 过程：对同一原图按 `PERTURBATION_MODE` 做在线变换（clean/noise/mask/blur）。
- 结果：teacher 取 pair 前项视图，student 取 pair 后项视图。
- 约束：单次实验固定一个 pair。
- 特殊约定：`clean-mask` 在实现中等价为 `mask-clean`（teacher=mask, student=clean）。

2. 数据集离线视图字段
- 条件：`use_image_perturbation_pairs=0` 且 `use_vcd_opsd=1`。
- 输入：数据集需有如 `problem_clean`、`problem_noise` 这类字段。
- 解析方式：`view_field_prefix + tag`。
- 若某样本缺失 pair 字段，会回退到 `good_view_field` / `bad_view_field`。

### 11.5 视图对怎么选择

通过 `pair_sampling_strategy` 控制：

- `first`：始终取 `view_pairs` 第一个，最稳定、便于复现。
- `random`：每个样本随机选一个可用 pair，增强多样性。
- `round_robin`：按样本索引轮换 pair，平衡覆盖。

在线扰动模式下：
- 当前实现固定单 pair，训练中等效使用 `first`。
- 启动脚本在 `USE_IMAGE_PERTURBATION_PAIRS=1` 时会自动固定 `PAIR_SAMPLING_STRATEGY=first`。

数据集离线字段模式下：
- 先过滤出“当前样本可用”的 pair，再按策略选择。

### 11.6 teacher 更新策略（与上面分支正交）

这两项是“teacher 参数来源”策略，可以与主分支组合：

- `fixed_teacher=1`：teacher 用初始策略（LoRA 关闭）固定监督。
- `use_ema_teacher=1`：teacher 用 student 参数的 EMA。

限制：
- `fixed_teacher=1` 需要 `use_peft=1`。
- `use_ema_teacher=1` 与 `fixed_teacher=1` 互斥。

---

如果你继续扩展新的扰动类型（如 occlusion grid、JPEG、color jitter），建议仅在 `data_collator.py` 新增 perturb tag 映射，训练主干可保持不变。



## 2024-4-12更新

1. 训练启动脚本大幅增强（核心都在 run_opsd_vcd_debug_4gpu.sh）：  
- 强制在 opsd conda 环境运行：run_opsd_vcd_debug_4gpu.sh  
- 默认输出到数据盘路径 output，避免系统盘爆满：run_opsd_vcd_debug_4gpu.sh  
- 默认 300 steps、每步日志、checkpoint 每 50 步、最多保留 3 个、只存模型：run_opsd_vcd_debug_4gpu.sh  
- dataloader workers 和 pin_memory 参数已加：run_opsd_vcd_debug_4gpu.sh  
- NCCL 修复：优先注入 compat libcuda 路径，避免 NCCL 初始化 SIGSEGV：run_opsd_vcd_debug_4gpu.sh  
2. 训练指标增强（WandB 可见）：  
- 新增生成吞吐和时延指标：gen_tokens_per_sec、gen_total_tokens、gen_avg_completion_len、gen_elapsed_sec  
- 位置：opsd_trainer.py 和 opsd_trainer.py  
3. 生成效率修复：  
- collator padding 改为 left，去掉 decoder-only 的 right-padding 低效路径  
- 位置：data_collator.py  
4. 分布式稳定性防护（之前已加）：  
- 分布式时自动关闭 gradient_checkpointing 以避开冲突：opsd_train.py  
- 分布式下避免 device_map auto 路径冲突：opsd_train.py  
5. 运维层面做过的处理：  
- 清理过大 checkpoint 释放空间（避免 No space left on device）  
- 多次重启验证后，已确认当前可在 opsd 环境下走 NCCL 路径启动。  


## 2026-04-13 本次训练结果（run: 94p6ua1j）

### 运行配置（关键项）

- 分布式：4 卡，`ddp_backend=nccl`
- 数据集：`derek-thomas/ScienceQA`（`validation` split）
- 训练步数：`max_steps=300`（实际完成到 step 300）
- batch：`per_device_train_batch_size=2`，`gradient_accumulation_steps=1`（等效全局 batch=8）
- 长度：`max_length=1024`，`max_completion_length=64`
- 保存策略：每 50 step 保存，`save_total_limit=3`，`save_only_model=true`

### 训练结果（WandB summary）

- `train/global_step`: 300
- `train_runtime`: 2276.9033 s（约 37.95 分钟）
- `train_steps_per_second`: 0.132
- `train_samples_per_second`: 1.054
- `train_loss`（全程平均）: 0.031968726714452105
- 最后一步（step 300）：
  - `train/loss`: 0.0208
  - `train/loss_total`: 0.017333984375
  - `train/loss_distill`: 0.017333984375
  - `train/grad_norm`: 2.2620441913604736
  - `train/learning_rate`: 6.666666666666667e-09
  - `train/gen_tokens_per_sec`: 78.63725228985366
  - `train/gen_avg_completion_len`: 64
  - `train/gen_total_tokens`: 128
  - `train/gen_elapsed_sec`: 1.6277272701263428

### 产物落盘

- 最终模型目录：`output/opsd_debug_4gpu/opsd_vcd_debug_4gpu`
- 最终 checkpoint：`checkpoint-300`（同时保留 `checkpoint-250`、`checkpoint-200`）
- 最终权重：
  - `model-00001-of-00002.safetensors`（4.7G）
  - `model-00002-of-00002.safetensors`（3.0G）
- 生成样本日志：`output/opsd_debug_4gpu/opsd_vcd_debug_4gpu/generations` 共 59 个文件（按 step 间隔保存）

### 备注

- 本次 run 从 metadata 可见 CUDA 版本为 12.8，GPU 为 NVIDIA vGPU-48GB（4 张）。
- 本次训练已完整跑到 `should_training_stop=true`，并成功写出 step 300 checkpoint 与最终模型文件。

如果你要，我可以下一步直接给你一条“安全清理命令”，只删旧 logs 和旧 wandb run，不动当前正在跑的最新 run。

## 2026-04-14 今日工作汇总（30-step 快速 + 1000-step 全训练 + POPE + MME + MMHal）

### A. 新增评测能力

1. MME 数据自动准备
- `scripts/prepare_mme_data.sh`
- 自动下载并解压到：`data/MME/MME_Benchmark_release_version/MME_Benchmark/...`

2. MME 单模型评测（支持 4 卡并行）
- `eval/evaluate_mme_qwen25vl.py`
- `scripts/run_mme_eval.sh`
- 支持 `PARALLEL_GPUS=4` 的 chunk 并行（每卡一个 chunk，结束后自动合并分数）

3. MME baseline vs ours 一键对照
- `scripts/run_mme_compare.sh`
- 同一评测逻辑连续跑两次：
  - baseline（`MODEL_PATH=base`，`BASE_MODEL_PATH=none`）
  - ours（`MODEL_PATH=adapter`，`BASE_MODEL_PATH=base`）
- 自动生成差值表：`ours - baseline`

### B. 1000-step 全训练流程（单 teacher）

- 训练目录：`output/opsd_full_4gpu/opsd_vcd_single_teacher_e1_cap1000_nccl_20260414_020000`
- 关键 checkpoint：`checkpoint-1000`
- 训练状态文件：`checkpoint-1000/trainer_state.json`
- 关键状态：
  - `global_step=1000`
  - `epoch=0.6285`
  - 最后一步（step 1000）：
    - `loss=-0.0125`
    - `loss_distill=-0.008972`
    - `gen_tokens_per_sec=42.97`

### C. 30-step 快速训练测评（单 teacher，sanity）

- 快速训练目录：`output/opsd_debug_4gpu/opsd_vcd_quick30_clean_noise_20260413_193753`
- 关键 checkpoint：`checkpoint-30`
- 训练状态文件：`checkpoint-30/trainer_state.json`
- 关键状态：
  - `global_step=30`
  - `epoch=0.0565`
  - 最后一步（step 30）：
    - `loss=0.0216`
    - `loss_distill=0.038086`
    - `gen_tokens_per_sec=68.51`
- 快速生成日志：`generations/generations_step_{5,10,15,20,25}.json`

可用于快速 POPE 复现的模型路径：
- `output/opsd_debug_4gpu/opsd_vcd_quick30_clean_noise_20260413_193753/checkpoint-30`

### D. POPE 官方口径结果（baseline vs ours）

- 输出目录：`output/eval_pope_official_e1_cap1000_nccl_20260414_020000`
- 汇总文件：`output/eval_pope_official_e1_cap1000_nccl_20260414_020000/pope_summary.md`

平均指标（3 split）：
- baseline：`avg_accuracy=0.8789`, `avg_f1=0.8677`, `avg_precision=0.9577`, `avg_recall=0.7933`
- ours：`avg_accuracy=0.8804`, `avg_f1=0.8698`, `avg_precision=0.9564`, `avg_recall=0.7978`

按 split 的 Accuracy：
- random：baseline `0.8910` -> ours `0.8930`
- popular：baseline `0.8790` -> ours `0.8807`
- adversarial：baseline `0.8667` -> ours `0.8677`

### E. MME 最新对照结果（4 卡并行）

- 输出根目录：`output/eval_mme_latest_20260414_compare_4gpu`
- ours 汇总：`output/eval_mme_latest_20260414_compare_4gpu/ours/mme_summary.md`
- baseline 汇总：`output/eval_mme_latest_20260414_compare_4gpu/baseline/mme_summary.md`
- 差值表：`output/eval_mme_latest_20260414_compare_4gpu/mme_compare_ours_vs_baseline.md`

总分对照（Ours - Baseline）：
- Total: `2222.09 - 2225.29 = -3.19`
- Perception: `+7.52`
- Cognition: `-10.71`

### F. 复现命令（MME baseline + ours）

```bash
source /etc/network_turbo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opsd

OUTPUT_ROOT=/root/autodl-tmp/opsd/output/eval_mme_latest_20260414_compare_4gpu \
BASELINE_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
OURS_MODEL_PATH=/root/autodl-tmp/opsd/output/opsd_full_4gpu/opsd_vcd_single_teacher_e1_cap1000_nccl_20260414_020000 \
OURS_BASE_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
PROCESSOR_PATH=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
MME_ROOT=/root/autodl-tmp/opsd/data/MME \
BATCH_SIZE=8 PARALLEL_GPUS=4 \
bash scripts/run_mme_compare.sh
```

### G. 清理建议

如需只保留最新一轮 MME 结果：

```bash
rm -rf output/eval_mme*
```

### H. MMHal 最新对照结果（4 卡并行）

- 输出根目录：`output/eval_mmhal_compare_latest_20260414_4gpu`
- ours 汇总：`output/eval_mmhal_compare_latest_20260414_4gpu/ours/mmhal_summary.json`
- baseline 汇总：`output/eval_mmhal_compare_latest_20260414_4gpu/baseline/mmhal_summary.json`
- 差值表：`output/eval_mmhal_compare_latest_20260414_4gpu/mmhal_compare_ours_vs_baseline.md`

本次为生成侧统计对比（官方 MMHal 分数依赖 GPT-4 judge API）：

- 样本数：ours `96`，baseline `96`
- 空回答：ours `0`，baseline `0`
- 平均回答词数：baseline `17.5208` -> ours `18.1042`（`+0.5833`）
- 平均回答字符数：baseline `149.9062` -> ours `152.8125`（`+2.9062`）

### I. 复现命令（MMHal baseline + ours）

```bash
source /etc/network_turbo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate opsd

OUTPUT_ROOT=/root/autodl-tmp/opsd/output/eval_mmhal_compare_latest_20260414_4gpu \
MMHAL_ROOT=/root/autodl-tmp/opsd/data/MMHal-Bench \
BASELINE_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
OURS_MODEL_PATH=/root/autodl-tmp/opsd/output/opsd_full_4gpu/opsd_vcd_single_teacher_e1_cap1000_nccl_20260414_020000 \
OURS_BASE_MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
PROCESSOR_PATH=/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
BATCH_SIZE=8 PARALLEL_GPUS=4 \
bash scripts/run_mmhal_compare.sh
```