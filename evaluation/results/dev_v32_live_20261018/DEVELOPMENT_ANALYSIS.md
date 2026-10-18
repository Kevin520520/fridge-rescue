# Development v3.2 analysis

The 24 reviewed development cases ran once per arm before the formal run.

| Metric | Single-prompt | Agentic |
|---|---:|---:|
| Attempts | 24 | 24 |
| Provider or output errors | 0 | 0 |
| Common success | 100% | 91.7% |
| Returned-output hard-constraint pass | 100% | 100% |
| Applicable all-input hit | 100% | 85.7% |
| Mean latency | 3710.2 ms | 1227.4 ms |
| Valid corpus ID rate | not applicable | 100% |
| Mean selection regret | not applicable | 0.000 |
| Corpus no-result accuracy | not applicable | 100% |

D07, D10, and D16 are reviewed partial-match cases whose best admitted local
recipes do not use every input. D20 and D21 are reviewed corpus gaps, while D22
is a contradictory request. The observed no-result behavior therefore matches
the corpus labels. No code change was required after the development run.
