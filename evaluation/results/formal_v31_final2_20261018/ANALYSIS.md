# Formal v3.1 result analysis

This analysis interprets the frozen 20-case, two-arm, three-repetition run in
this directory. It was written after the run. No formal result was replaced or
discarded because it was unfavorable.

## Main result

| Metric | Single-prompt | Agentic |
|---|---:|---:|
| Attempts | 60 | 60 |
| Provider or output errors | 2 | 1 |
| Common success | 96.7% | 93.3% |
| Returned-output hard-constraint pass | 100% | 100% |
| Applicable all-input hit | 94.7% | 81.5% |
| Mean latency | 3814.6 ms | 1335.8 ms |
| Valid corpus ID rate | not applicable | 100% |

The result does not show universal Agentic superiority. Single-prompt generation
can invent a suitable new recipe and therefore uses all requested ingredients
more often. Agentic output is restricted to the fixed 2,000-recipe corpus. Its
benefits in this run are provenance, deterministic screening, 100% valid selected
IDs, 100% hard-constraint compliance among returned recipes, and lower average
latency.

## Observed failures

- Single-prompt had two transient provider/output failures: F18 repetition 1
  and F06 repetition 3.
- Agentic had one output-validation failure: F13 repetition 3. The model added
  substitution notes outside the reviewed allowlist, did so again after the one
  permitted repair, and the workflow safely returned no recommendation rather
  than displaying unsupported text. The other two F13 repetitions succeeded.
- F18 (durian) is an intentional corpus-gap case. Single-prompt can invent a
  durian recipe; Agentic correctly reports that the fixed local corpus has no
  admitted candidate.
- F19 is a contradictory request in which the only available ingredient is also
  excluded. Both arms correctly reject it.

## Additional limitations found

1. Corpus coverage is the main ceiling on Agentic input utilization. A safe,
   valid partial match may be available even when no stored recipe uses every
   ingredient.
2. A rescue candidate may still require up to eight extra non-pantry ingredients.
   The interface must keep showing the missing-ingredient list so this trade-off
   remains visible.
3. Conservative vegan and vegetarian labels are too sparse for broad product
   claims. They remain backend research fields rather than promoted UI filters.
4. The source has no dependable time, rating, review-count, serving, or yield
   fields. These values remain null and are not inferred by the LLM.
5. `meal_type` is a soft prompt context, not a deterministic corpus filter.
6. Twenty cases with three repetitions support descriptive course-project
   comparison, not a universal or statistically conclusive product claim.

## Release decision

The release keeps the bounded-repair and safe-failure behavior. It does not add
an after-the-fact exception for F13 because that would tune the code on the
formal test set and could allow unsupported substitution advice. Future work can
test a separately designed deterministic sanitizer on a new held-out set.

The two-tier retrieval policy was selected on the development set before this
formal run. The formal results are frozen and should be reported as measured.

After the formal run, one presentation-only bug was found during the live
deployment smoke test: the Local retrieval explanation counted assumed pantry
staples as ingredients the user explicitly entered. The UI wording now reports
requested-input utilization separately from the extra shopping list. Retrieval,
selection, validation, and all saved formal metrics were unchanged; the frozen
source manifest intentionally retains the exact pre-presentation-fix app hash.
