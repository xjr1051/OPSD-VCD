# MMHal Ours vs Baseline

This document includes both generation-side statistics and official MMHal judge results.

## Official MMHal Judge (rerun on yunwu.ai)

Judge config: model = gpt-4o, api_base = https://yunwu.ai/v1, num_samples = 96

| Metric | Ours | Baseline | Delta (Ours-Baseline) |
|---|---:|---:|---:|
| avg_score (higher is better) | 1.8542 | 1.8646 | -0.0104 |
| hallucination_rate (lower is better) | 0.6458 | 0.6667 | -0.0208 |

### Judge Avg Score by Question Type

| Question Type | Ours | Baseline | Delta (Ours-Baseline) |
|---|---:|---:|---:|
| adversarial | 2.2500 | 2.0000 | +0.2500 |
| attribute | 1.3333 | 2.0000 | -0.6667 |
| comparison | 1.3333 | 1.4167 | -0.0833 |
| counting | 2.0000 | 1.6667 | +0.3333 |
| environment | 2.1667 | 2.1667 | +0.0000 |
| holistic | 0.5833 | 0.4167 | +0.1667 |
| other | 2.3333 | 2.1667 | +0.1667 |
| relation | 2.8333 | 3.0833 | -0.2500 |

## Generation-side Comparison

| Metric | Ours | Baseline | Delta (Ours-Baseline) |
|---|---:|---:|---:|
| num_samples | 96.0000 | 96.0000 | +0.0000 |
| avg_response_words | 18.1042 | 17.5208 | +0.5833 |
| avg_response_chars | 152.8125 | 149.9062 | +2.9062 |
| empty_answers | 0.0000 | 0.0000 | +0.0000 |

| Question Type | Ours Count | Baseline Count | Count Delta | Ours Avg Words | Baseline Avg Words | Word Delta |
|---|---:|---:|---:|---:|---:|---:|
| adversarial | 12 | 12 | +0 | 15.6667 | 20.0833 | -4.4167 |
| attribute | 12 | 12 | +0 | 17.4167 | 9.0833 | +8.3333 |
| comparison | 12 | 12 | +0 | 24.4167 | 23.1667 | +1.2500 |
| counting | 12 | 12 | +0 | 4.3333 | 4.2500 | +0.0833 |
| environment | 12 | 12 | +0 | 14.6667 | 10.8333 | +3.8333 |
| holistic | 12 | 12 | +0 | 36.5833 | 37.0833 | -0.5000 |
| other | 12 | 12 | +0 | 15.5000 | 16.2500 | -0.7500 |
| relation | 12 | 12 | +0 | 16.2500 | 19.4167 | -3.1667 |
