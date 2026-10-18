# Corpus v3.1 curation audit trail

This file records the deterministic release pipeline and its quality gates. It
does not claim that every source recipe has been kitchen-tested.

## Source contract

- Dataset: Epicurious Recipes (13k)
- Repository: https://github.com/josephrmartinez/recipe-dataset
- Pinned revision: `c5a1e8dab69d46841d914be1f20cca6fbe4b6260`
- Pinned file: `13k-recipes.csv`
- Local raw file: `data/raw/epicurious_recipes.csv`
- Raw rows: `13,501`
- Raw size: `26,595,455` bytes
- Raw SHA-256: `d0e3977da7c1dc287ed75d6770f4064a7caeaa20ef6f64708b730e02f4d5f17a`
- Source and derived-corpus license: CC BY-SA 3.0

The downloader rejects a file whose size or SHA-256 differs from this pinned
source contract.

## Deterministic release pipeline

```text
13,501 pinned source rows
  -> parse all ingredient and instruction list fields
  -> reject any record containing a confirmed corrupt, equipment-only,
     metadata, unresolved component, or structurally incomplete line
  -> validate amounts, units, conversions, text integrity, and source fidelity
  -> deduplicate complete recipe records
  -> 7,578 eligible records
  -> fixed-seed category/complexity stratified selection
  -> 2,000 release artifacts
```

The pipeline rejects the whole recipe when a confirmed bad line is present. It
does not hide that line while retaining the recipe. For all selected records,
the source, original, and normalized ingredient arrays each contain exactly
`20,997` lines; the recorded removed-line count is `0`.

Each retained recipe has at least three ingredient lines, at least two
instruction steps, and at least 60% source lines with an explicit measurement.
The released corpus has 93.47% mean measurement coverage.

## Quantity and unit policy

- `ingredient_lines_original` preserves every source-provided ingredient line.
- Missing amounts or units are never invented, inferred, or filled by the LLM.
- A malformed amount or unit rejects the complete source record.
- Legitimate wording such as `salt to taste` may remain and is counted as a
  line without an explicit measurement.
- Only unambiguous formatting is normalized, such as restoring spacing in a
  joined mixed fraction. The release contains 18 such corrections across 16
  records; the amount itself is unchanged.
- Unit grammar, glued amount/unit tokens, fraction formatting, cup/fluid-ounce
  consistency, implausible large unitless amounts, and common OCR defects are
  checked before selection and again by the full-corpus audit.

## Release result

- Selected rows: `2,000`
- Unique recipe IDs: `2,000`
- Unique source-record IDs: `2,000`
- Selected ingredient lines: `20,997`
- Removed or hidden ingredient lines: `0`
- Measurement coverage: minimum `60.0%`, mean `93.47%`, maximum `100%`
- Curated Parquet SHA-256:
  `1fbeeaa09544940884e64dc94a9b3666658a5956790c7f36487f26892371e690`
- Rules SHA-256:
  `b1622d52c4aa886c90480cdba384afa85d07a0ef54db402367f38e0ed3a19cc0`

Machine-readable rejection counts and stratum totals are in
`curation_summary.json`. The current rules include 743 reviewed source-record
exclusions confirmed during iterative adversarial review.

## Full-corpus audit

Run:

```powershell
.\.venv\Scripts\python.exe scripts\audit_curated_corpus.py `
  --corpus data\curated\recipes_2000.parquet `
  --summary data\curated\curation_summary.json `
  --output-dir data\curated\audits\v31_20261018
```

Evidence is stored in `data/curated/audits/v31_20261018/`:

- `corpus_audit.json`: release result and source contract
- `record_audit.csv`: one audit row per released recipe
- `flagged_records.csv`: hard failures or review warnings

For the released corpus SHA-256 above, all 2,000 records passed with zero
global failures, zero hard failures, zero review warnings, and no disallowed
split-quantity columns.

An independent rebuild reproduced all four declared release files byte for
byte. The only non-deterministic summary field is elapsed runtime.

## Rebuild commands

```powershell
.\.venv\Scripts\python.exe scripts\download_source.py
.\.venv\Scripts\python.exe scripts\curate_corpus.py --output-dir data\curated
.\.venv\Scripts\python.exe scripts\audit_curated_corpus.py
.\.venv\Scripts\python.exe scripts\check_setup.py --check-rebuild
```

Only the files listed in the current curation summary and audit directory are
release evidence for Corpus v3.1.
