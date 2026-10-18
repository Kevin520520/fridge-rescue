# Formal v3.2 regression analysis

This run compares the frozen v3.2 application with the same 20 reviewed cases
used for v3.1. Each arm ran three times, producing 120 attempts. Because the
cases were reused, this is a version-to-version regression comparison rather
than a new blind test. No runtime behavior or evaluated configuration value
was changed after this run. Later release cleanup removed one obsolete UI test
and rewrote non-executable comments and documentation; the local full result
directory retains the exact frozen files and source manifest from the experiment.

## Current result

| Metric | Single-prompt | Agentic |
|---|---:|---:|
| Attempts | 60 | 60 |
| Provider or output errors | 1 | 0 |
| Common success | 98.3% | 95.0% |
| Returned-output hard-constraint pass | 100% | 100% |
| Applicable all-input hit | 98.2% | 83.3% |
| Mean latency | 3968.5 ms | 1308.3 ms |
| Valid corpus ID rate | not applicable | 100% |
| Mean selection regret | not applicable | 0.000 |
| Corpus no-result accuracy | not applicable | 100% |

## Comparison with v3.1

| Agentic metric | v3.1 | v3.2 | Change |
|---|---:|---:|---:|
| Provider or output errors | 1 | 0 | -1 |
| Common success | 93.3% | 95.0% | +1.7 pp |
| Applicable all-input hit | 81.5% | 83.3% | +1.8 pp |
| Mean latency | 1335.8 ms | 1308.3 ms | -27.5 ms |
| Valid corpus ID rate | 100% | 100% | unchanged |
| Hard-constraint pass | 100% | 100% | unchanged |
| Mean selection regret | 0.000 | 0.000 | unchanged |

The v3.2 selector receives only the three strongest results from the
deterministically ranked Top-10 retrieval pool. This prevents it from choosing
a much weaker lower-ranked candidate while preserving up to three menu choices.
Explicit pantry items entered by the user also count toward input utilization;
pantry assumptions still do not create false user-input matches in the UI.

## Interpretation

The result does not support a claim that Agentic is universally better.
Single-prompt generation uses all requested ingredients more often because it
can create a new recipe. Agentic is intentionally restricted to the fixed
2,000-recipe corpus. Its measured strengths are traceable recipe IDs, original
source quantities and steps, deterministic hard-constraint checks, safe
no-result behavior, and lower average latency in this run.

F12, F14, and F15 reach only two-thirds input utilization because the filtered
local corpus contains no admitted full-input recipe. Selection regret remains
zero, so changing the selector cannot remove those corpus gaps. F18 is a
reviewed corpus-gap case and Agentic consistently returns no result. F19 is a
contradictory request and both arms correctly return no result. The only model
error was one Single-prompt provider/output failure on F07.

## Release decision

The code is accepted without further tuning on the formal cases. The remaining
input-coverage gap is a documented corpus limitation, not evidence of an
invalid ID, hard-constraint failure, or lower-ranked Agentic selection. The
saved aggregate report, case comparison, repetitions, frozen inputs, hashes,
configuration, rules, and prompts are the authoritative v3.2 experiment
evidence.
