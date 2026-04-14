# HallusionBench Baseline vs Ours

- Baseline: `Qwen2.5-VL-3B-Instruct`
- Ours: `opsd_vcd_single_teacher_e1_cap1000_nccl_20260414_020000`

| Metric | Baseline | Ours | Delta (Ours-Baseline) |
|---|---:|---:|---:|
| qAcc | 28.3516 | 28.1319 | -0.2197 |
| fAcc | 36.9942 | 35.8382 | -1.1560 |
| Easy Acc | 60.6593 | 60.4396 | -0.2197 |
| Hard Acc | 51.1628 | 50.2326 | -0.9302 |
| aAcc | 60.7617 | 59.9646 | -0.7971 |
| VD Acc | 54.4839 | 53.9763 | -0.5076 |
| VS Acc | 67.6580 | 66.5428 | -1.1152 |
| Overall Acc | 60.7617 | 59.9646 | -0.7971 |

## Sources

- Baseline summary: `output/eval_hallusionbench_baseline_20260414_4gpu/hallusionbench_summary.json`
- Ours summary: `output/eval_hallusionbench_latest_20260414_4gpu/hallusionbench_summary.json`
