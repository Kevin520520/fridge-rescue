# Formal case review record

- Case file: `evaluation/final_cases.csv`
- Source proposal: `evaluation/proposed_final_cases_v2.csv`
- Case count: 20
- Split: `final`
- Reviewer: project group member (user confirmation in the project conversation)
- Confirmation date: 2026-10-17
- Confirmation text: `确认 v2 的 20 条正式测试案例`

The reviewed file preserves the twenty v2 requests, labels, and notes. The
only planned change from the proposal is `label_status=reviewed`, recording the
human confirmation required before the formal API experiment.

The formal comparison must retain all errors and outputs. If a genuine program
defect is discovered from these final cases, the affected run must be treated
as invalid development evidence; after a fix, a new unseen final set is needed.
