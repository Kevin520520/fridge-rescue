# Evaluation aggregate

Completed repetitions: 3.

Metrics are pooled over every saved run. Automatic text screens are diagnostics and should not be described as human judgments.

| Arm | Runs | Errors | Common success | Hard constraints | All-input hit | Mean latency ms | Tokens in/out | Estimated cost USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_a | 60 | 1 | 0.983 | 1.000 | 0.982 | 3968.5 | None/None | n/a |
| agentic | 60 | 0 | 0.950 | 1.000 | 0.833 | 1308.3 | 77655/8365 | n/a |

Use `case_comparison.csv` to inspect repeated failures, constraint flags, retrieval quality and selection regret before changing code.
