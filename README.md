# PRISM-OPSD: Privileged Visual Self-Distillation for Grounded VLMs

本仓库是一个新的实验研究项目：在 OPSD 代码基座上，将 VCD 的“原图-扰动图对比”思想迁移到训练阶段，用于降低 VLM 幻觉。

## 1. 核心想法

VCD 告诉我们：对同一图像构造扰动视图，并与 clean 视图做对比解码，可以减轻 hallucination。

本项目把这一思想改造成训练期范式：

- student 在较弱/扰动视图条件下 on-policy 采样轨迹；
- teacher 在较强/更事实视图条件下给出监督；
- teacher 分布通过“teacher-view vs student-view”对比 logits 构造。

适配的 teacher-student 视图对包括但不限于：

- clean-noise
- mask-clean
- clean-blur
- crop-clean

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
- `--view_pairs`：teacher-student 视图对列表，例：`clean-noise,mask-clean`。
- `--view_field_prefix`：视图字段前缀，默认 `problem_`。
- `--pair_sampling_strategy`：多视图对采样策略：`random | first | round_robin`。
- `--use_image_perturbation_pairs`：在 collator 内对图像在线扰动并生成视图对。
- `--image_field`：原始图像列名，默认 `image`。
- `--image_token`：多模态提示中的图像占位符，默认 `<image>`。
- `--noise_std` / `--mask_ratio` / `--blur_radius`：在线扰动参数。
- `--use_multimodal_processor`：加载 `AutoProcessor`（图片训练必需）。
- `--use_privileged_visual_teacher`：启用 teacher-only 的 privileged visual evidence 条件。
- `--privileged_visual_field`：数据中的 `z^*` 字段名。

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
bash scripts/run_opsd_vcd_skeleton.sh
```

脚本已包含：

- `--use_vcd_opsd`
- `--use_multimodal_processor`
- `--use_image_perturbation_pairs`
- `--use_privileged_visual_teacher`
- `--view_pairs clean-noise,mask-clean`
- `--pair_sampling_strategy random`

## 7. 推荐实验矩阵

建议最小实验集：

- alpha 扫描：$\alpha \in \{0.25, 0.5, 1.0, 2.0\}$
- pair 组合：
  - clean-noise
  - mask-clean
  - clean-noise,mask-clean（联合）
- 采样策略：
  - random
  - round_robin

对照组：

- OPSD baseline（关闭 `--use_vcd_opsd`）
- PRISM-OPSD（开启 `--use_vcd_opsd` + `--use_privileged_visual_teacher`）

## 8. 评估建议

建议至少报告以下指标：

- 任务准确率（task accuracy）
- 幻觉率/事实一致性（hallucination or faithfulness metric）
- 稳定性统计（loss 波动、长度分布、梯度范数）

## 9. 重要说明

- 该实现是“训练期融合骨架”，用于将视图对比偏好蒸馏进参数。
- 当前版本已经支持在 collator 中在线生成 clean/noise/mask/blur 视图对。
- `use_image_perturbation_pairs=True` 时，当前不支持 `use_vllm`，请使用 transformers 生成路径。
- 如使用特定 VLM（如 Qwen-VL/LLaVA），请确保 `image_token` 与模型模板一致。

## 10. 仓库结构（核心）

- `opsd_train.py`：训练入口与参数定义。
- `data_collator.py`：teacher-student 视图对组装。
- `opsd_trainer.py`：on-policy 轨迹 + token-level self-distillation 蒸馏。
- `scripts/run_opsd_vcd_skeleton.sh`：示例启动脚本。

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

如果你要，我可以下一步直接给你一条“安全清理命令”，只删旧 logs 和旧 wandb run，不动当前正在跑的最新 run。