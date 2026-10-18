# Formal v3.3 regression analysis

This release changes retrieval presentation and shopping-burden handling without
changing the pinned Epicurious source or the 2,000-recipe corpus. The same 20
reviewed final cases were run three times in each arm, producing 120 API-backed
attempts. Because the cases were reused, this is a version regression test, not
a new blind test.

## Release changes under test

- Matched ingredient words or phrases are highlighted inline; the full source
  line and the former `In your fridge` badge are no longer highlighted.
- Deterministic ranking now states and applies the exact priority order:
  maximize user-input utilization, then minimize missing non-pantry ingredient
  lines, then maximize recipe-side coverage.
- Agentic output validation enforces the first two priorities, and hydrated
  recommendations are displayed in the same order.
- Pantry detection is whole-line and conservative. Compound products such as
  low-salt broth, rose water, coconut water, soda water, water chestnuts, and
  salt-packed capers remain visible as required extra ingredients.

## Automated verification

- Full unit and integration suite: 436 passed.
- Curated corpus audit: 2,000/2,000 rows passed, with zero hard failures and
  zero review warnings.
- Forty-four-case offline retrieval regression: zero stale corpus-result
  labels and zero violations of the utilization-then-shopping-burden order.
- The English and reviewed Chinese forms of the egg-and-tomato request returned
  identical ranked candidate IDs.

## Formal result

| Metric | Single-prompt | Agentic |
|---|---:|---:|
| Attempts | 60 | 60 |
| Provider or output errors | 0 | 0 |
| Common success | 100.0% | 95.0% |
| Returned-output hard-constraint pass | 100% | 100% |
| Applicable all-input hit | 98.2% | 83.3% |
| Mean latency | 3963.6 ms | 1296.7 ms |
| Valid corpus ID rate | not applicable | 100% |
| Mean selection regret | not applicable | 0.000 |
| Corpus no-result accuracy | not applicable | 100% |

The result does not support a claim that Agentic is universally better.
Single-prompt generation can create a new recipe and therefore uses every input
more often. Agentic is restricted to a fixed corpus, but it provides traceable
recipe IDs, source-preserved quantities and steps, deterministic constraint
checks, transparent shopping burden, and safe no-result behavior.

F12, F14, and F15 remain corpus-coverage gaps: their best admitted candidates
use two-thirds of the requested ingredients. Selection regret is zero, so the
selector did not skip a better retrieved match. F18 is the reviewed corpus-gap
case and consistently returns no result. F19 is contradictory and both arms
correctly return no result.

## Release decision

No code or evaluated configuration was changed after the formal run. The
complete local run retains the three repetition-level attempt logs. The compact
public release keeps the frozen inputs, hashes, aggregate report, and case
comparison needed to review the authoritative v3.3 results without publishing
the larger attempt logs.
