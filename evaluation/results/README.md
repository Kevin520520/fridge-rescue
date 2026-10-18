# Current evaluation evidence

Corpus v3.1 uses the pinned Epicurious source and curated corpus SHA-256
`1fbeeaa09544940884e64dc94a9b3666658a5956790c7f36487f26892371e690`.
Only the following directories are current application evidence:

- `dev_v32_live_20261018/`: 24-case, two-arm development run used before the
  v3.2 code freeze.
- `formal_v32_20261018/`: 20-case, two-arm, three-repetition regression
  comparison used for the report.

Other experiment iterations are not part of the current release and must not
be quoted as current results.

The local formal directory retains every raw attempt, frozen
cases/config/prompts/rules, source manifest, aggregate table, case comparison,
and the post-run analysis. The public repository keeps a compact evidence set
without the large raw attempt files.
