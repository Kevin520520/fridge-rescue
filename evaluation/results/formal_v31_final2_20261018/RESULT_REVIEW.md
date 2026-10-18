# Evaluation aggregate

Completed repetitions: 3.

Metrics are pooled over every saved run. Automatic text screens are diagnostics and should not be described as human judgments.

| Arm | Runs | Errors | Common success | Hard constraints | All-input hit | Mean latency ms | Tokens in/out | Estimated cost USD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_a | 60 | 2 | 0.967 | 1.000 | 0.947 | 3814.6 | None/None | n/a |
| agentic | 60 | 1 | 0.933 | 1.000 | 0.815 | 1335.8 | 180711/8599 | n/a |

Use `case_comparison.csv` to inspect repeated failures, constraint flags, retrieval quality and selection regret before changing code.
