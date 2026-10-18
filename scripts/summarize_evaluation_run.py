"""Pool completed repetitions from one Agentic-versus-baseline experiment.

This script is offline. It reads the complete ``comparison_report.json`` files
created by ``run_final_evaluation.py`` and writes reproducible aggregate and
per-case tables. It never reads an API key or calls a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from fridge_rescue.evaluation import CaseEvaluation, summarize_results


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _optional_mean(values: Iterable[float | int | None]) -> float | None:
    known = [float(value) for value in values if value is not None]
    return fmean(known) if known else None


def _optional_sum(values: Iterable[float | int | None]) -> float | int | None:
    materialized = list(values)
    if not materialized or any(value is None for value in materialized):
        return None
    return sum(value for value in materialized if value is not None)


def _load_reports(run_dir: Path) -> tuple[list[CaseEvaluation], list[dict[str, object]]]:
    report_paths = sorted(run_dir.glob("repetition_*/comparison_report.json"))
    if not report_paths:
        raise ValueError(f"No completed repetition reports found in {run_dir}")

    all_results: list[CaseEvaluation] = []
    manifests: list[dict[str, object]] = []
    expected_identity: tuple[object, ...] | None = None
    expected_keys: set[tuple[str, str]] | None = None
    for path in report_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise TypeError(f"Invalid metadata in {path}")
        identity = (
            payload.get("case_set_sha256"),
            metadata.get("corpus_sha256"),
            metadata.get("model"),
            metadata.get("arms"),
        )
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise ValueError("Repetition reports do not share one frozen experiment identity")

        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise TypeError(f"Missing results in {path}")
        results = [CaseEvaluation.model_validate(item) for item in raw_results]
        keys = [(item.runner_name, item.case_id) for item in results]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate runner/case rows in {path}")
        key_set = set(keys)
        if expected_keys is None:
            expected_keys = key_set
        elif key_set != expected_keys:
            raise ValueError("Repetition reports do not contain the same runner/case rows")

        attempts_path = path.with_name("attempts.jsonl")
        if not attempts_path.is_file():
            raise ValueError(f"Missing crash-resistant attempts log: {attempts_path}")
        attempt_rows = [
            json.loads(line)
            for line in attempts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        attempt_keys = {
            (str(item.get("runner_name", "")), str(item.get("case_id", "")))
            for item in attempt_rows
        }
        if len(attempt_rows) != len(results) or attempt_keys != key_set:
            raise ValueError(f"Attempts log does not match completed report: {attempts_path}")
        all_results.extend(results)
        manifests.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": _sha256(path),
                "attempts_path": attempts_path.relative_to(run_dir).as_posix(),
                "attempts_sha256": _sha256(attempts_path),
                "result_count": len(results),
            }
        )
    return all_results, manifests


def _summary_rows(results: Sequence[CaseEvaluation]) -> list[dict[str, object]]:
    runner_names = list(dict.fromkeys(result.runner_name for result in results))
    return [
        summarize_results(results, runner_name).model_dump(mode="json")
        for runner_name in runner_names
    ]


def _case_rows(results: Sequence[CaseEvaluation]) -> list[dict[str, object]]:
    keys = sorted({(result.case_id, result.runner_name) for result in results})
    rows: list[dict[str, object]] = []
    for case_id, runner_name in keys:
        selected = [
            result
            for result in results
            if result.case_id == case_id and result.runner_name == runner_name
        ]
        stages = Counter(
            result.failure_stage for result in selected if result.failure_stage is not None
        )
        rows.append(
            {
                "case_id": case_id,
                "runner_name": runner_name,
                "repetitions": len(selected),
                "success_runs": sum(
                    result.prediction_status == "success" for result in selected
                ),
                "no_result_runs": sum(
                    result.prediction_status == "no_result" for result in selected
                ),
                "error_runs": sum(result.prediction_status == "error" for result in selected),
                "common_success_rate": _optional_mean(
                    float(result.common_success)
                    if result.common_success is not None
                    else None
                    for result in selected
                ),
                "hard_constraint_failure_runs": sum(
                    result.common_hard_constraint_pass_rate is not None
                    and result.common_hard_constraint_pass_rate < 1.0
                    for result in selected
                ),
                "retrieval_no_candidate_runs": sum(
                    result.retrieved_candidate_ids == [] for result in selected
                ),
                "retrieval_full_input_gap_runs": sum(
                    result.best_retrieved_input_utilization_rate is not None
                    and result.best_retrieved_input_utilization_rate < 1.0
                    and result.all_requested_ingredients_metric_applicable
                    for result in selected
                ),
                "selection_regret_runs": sum(
                    result.selection_input_utilization_regret is not None
                    and result.selection_input_utilization_regret > 0
                    for result in selected
                ),
                "repair_runs": sum((result.retry_count or 0) > 0 for result in selected),
                "validation_error_runs": sum(
                    bool(result.validation_errors) for result in selected
                ),
                "invalid_retrieval_id_runs": sum(
                    bool(result.invalid_retrieved_candidate_ids) for result in selected
                ),
                "all_requested_ingredients_hit_rate": _optional_mean(
                    float(result.all_requested_ingredients_used)
                    if result.all_requested_ingredients_metric_applicable
                    else None
                    for result in selected
                ),
                "mean_best_input_utilization": _optional_mean(
                    result.best_common_input_utilization_rate for result in selected
                ),
                "mean_latency_ms": _optional_mean(result.latency_ms for result in selected),
                "total_input_tokens": _optional_sum(
                    result.input_tokens for result in selected
                ),
                "total_output_tokens": _optional_sum(
                    result.output_tokens for result in selected
                ),
                "total_estimated_cost_usd": _optional_sum(
                    result.estimated_cost_usd for result in selected
                ),
                "mean_retrieved_candidate_count": _optional_mean(
                    result.retrieved_candidate_count for result in selected
                ),
                "mean_best_retrieved_input_utilization": _optional_mean(
                    result.best_retrieved_input_utilization_rate for result in selected
                ),
                "mean_selection_input_utilization_regret": _optional_mean(
                    result.selection_input_utilization_regret for result in selected
                ),
                "failure_stages_json": json.dumps(stages, sort_keys=True),
            }
        )
    return rows


def _markdown(summaries: Sequence[Mapping[str, object]], repetition_count: int) -> str:
    lines = [
        "# Evaluation aggregate",
        "",
        f"Completed repetitions: {repetition_count}.",
        "",
        (
            "Metrics are pooled over every saved run. Automatic text screens are "
            "diagnostics and should not be described as human judgments."
        ),
        "",
        (
            "| Arm | Runs | Errors | Common success | Hard constraints | "
            "All-input hit | Mean latency ms | Tokens in/out | Estimated cost USD |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| {runner_name} | {case_count} | {error_case_count} | {success} | "
            "{hard} | {ingredients} | {latency} | {input_tokens}/{output_tokens} | "
            "{cost} |".format(
                runner_name=summary["runner_name"],
                case_count=summary["case_count"],
                error_case_count=summary["error_case_count"],
                success=_format_metric(summary.get("common_success_rate")),
                hard=_format_metric(summary.get("common_hard_constraint_pass_rate")),
                ingredients=_format_metric(
                    summary.get("applicable_all_requested_ingredients_case_rate")
                ),
                latency=_format_metric(summary.get("mean_latency_ms"), digits=1),
                input_tokens=summary.get("total_input_tokens"),
                output_tokens=summary.get("total_output_tokens"),
                cost=_format_metric(summary.get("total_estimated_cost_usd"), digits=6),
            )
        )
    lines.extend(
        [
            "",
            (
                "Use `case_comparison.csv` to inspect repeated failures, constraint "
                "flags, retrieval quality and selection regret before changing code."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _format_metric(value: object, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def summarize_run(run_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    results, report_manifest = _load_reports(run_dir)
    summaries = _summary_rows(results)
    cases = _case_rows(results)
    repetition_count = len(report_manifest)

    _write_csv(run_dir / "aggregate_summary.csv", summaries)
    _write_csv(run_dir / "case_comparison.csv", cases)
    (run_dir / "aggregate_report.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "repetition_count": repetition_count,
                "reports": report_manifest,
                "summaries": summaries,
                "cases": cases,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "RESULT_REVIEW.md").write_text(
        _markdown(summaries, repetition_count), encoding="utf-8"
    )
    return summaries, cases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.run_dir.is_dir():
        print(f"[FAILED] Run directory not found: {args.run_dir}")
        return 1
    try:
        summaries, cases = summarize_run(args.run_dir)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        print(f"[FAILED] {error}")
        return 1
    print(
        f"Saved aggregate evaluation: {args.run_dir} "
        f"({len(summaries)} arms, {len(cases)} case-arm rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
