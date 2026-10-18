# Evaluation aggregate

Completed repetitions: 3.

Metrics are pooled over every saved run. Automatic text screens are diagnostics and should not be described as human judgments.

| Arm | Runs | Errors | Common success | Hard constraints | All-input hit | Mean latency ms | Tokens in/out | Estimated cost USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_a | 60 | 0 | 1.000 | 1.000 | 0.982 | 3963.6 | 45714/38461 | n/a |
| agentic | 60 | 0 | 0.950 | 1.000 | 0.833 | 1296.7 | 77424/8094 | n/a |

Use `case_comparison.csv` to inspect repeated failures, constraint flags, retrieval quality and selection regret before changing code.
