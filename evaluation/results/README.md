# Current evaluation evidence

Corpus v3.1 uses the pinned Epicurious source and curated corpus SHA-256
`1fbeeaa09544940884e64dc94a9b3666658a5956790c7f36487f26892371e690`.
Only the following directories are current release evidence:

- `v31_policy_scan_two_tier_20261018/`: deterministic development-set policy
  comparison used to choose the two-tier retrieval thresholds.
- `dev_v31_two_tier_live_20261018/`: 24-case live development run after the
  retrieval policy was selected.
- `formal_v31_final2_20261018/`: frozen 20-case, two-arm, three-repetition
  formal comparison used for the report.

All other result folders in the private development workspace are superseded
diagnostics, failed candidates, or historical Food.com v1/v2 evidence. They are
not part of the public release and must not be quoted as current results.

The formal directory retains every raw attempt, frozen cases/config/prompts/rules,
source manifest, aggregate table, case comparison, and the post-run analysis.
