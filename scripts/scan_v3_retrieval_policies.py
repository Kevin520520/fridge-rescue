"""Compare retrieval admission policies over the frozen v3 case files.

This is an offline diagnostic: it never calls an external model.  Candidate
relevance is measured with the same configured ingredient aliases and pantry
rules as the application.  A candidate is labelled low-input-relevance when it
uses less than half of the non-pantry ingredients supplied by the user.  The
report separately counts candidates below 25% recipe coverage and candidates
requiring more than six additional non-pantry ingredients, so a permissive
policy cannot appear better merely by admitting impractical recipes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from fridge_rescue.evaluation import EvaluationCase, load_cases_csv
from fridge_rescue.repository import DEFAULT_CURATED_CORPUS_PATH, ParquetRecipeRepository
from fridge_rescue.retrieval import (
    DeterministicRecipeRetriever,
    RetrievalPolicy,
    RetrievalRules,
    ingredient_match_details,
)
from fridge_rescue.settings import PROJECT_ROOT

DEFAULT_DEV_CASES = PROJECT_ROOT / "evaluation" / "dev_cases_v3.csv"
DEFAULT_FINAL_CASES = PROJECT_ROOT / "evaluation" / "final_cases_v3.csv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "evaluation" / "results" / "v3_policy_scan_20261018"
)
TOP_K = 10
LOW_RELEVANCE_CUTOFF = 0.5


@dataclass(frozen=True)
class PolicySpec:
    label: str
    minimum_coverage: float
    max_missing_ingredients: int
    rescue_minimum_coverage: float | None = None
    rescue_max_missing_ingredients: int | None = None


POLICIES = (
    PolicySpec("coverage_025_missing_6", 0.25, 6),
    PolicySpec("coverage_020_missing_7", 0.20, 7),
    PolicySpec("coverage_020_missing_8", 0.20, 8),
    PolicySpec("two_tier_020_7_rescue_015_8", 0.20, 7, 0.15, 8),
    PolicySpec("coverage_015_missing_8", 0.15, 8),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-cases", type=Path, default=DEFAULT_DEV_CASES)
    parser.add_argument("--final-cases", type=Path, default=DEFAULT_FINAL_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _validate_v3_cases(cases: Sequence[EvaluationCase]) -> None:
    timed = [case.case_id for case in cases if case.request.max_time_minutes is not None]
    if timed:
        raise ValueError("v3 cases must not use unavailable source time: " + ", ".join(timed))
    placeholders = [case.case_id for case in cases if case.label_status != "reviewed"]
    if placeholders:
        raise ValueError("v3 cases must be reviewed: " + ", ".join(placeholders))


def _unconstrained_case(case: EvaluationCase) -> EvaluationCase:
    request = case.request.model_copy(
        update={
            "excluded_ingredients": [],
            "unavailable_equipment": set(),
            "diet_preference": "none",
        }
    )
    return case.model_copy(update={"request": request})


def _case_rows(
    cases: Sequence[EvaluationCase],
    *,
    repository: ParquetRecipeRepository,
    rules: RetrievalRules,
    policy_spec: PolicySpec,
) -> list[dict[str, object]]:
    retriever = DeterministicRecipeRetriever(
        repository,
        rules=rules,
        policy=RetrievalPolicy(
            minimum_coverage=policy_spec.minimum_coverage,
            max_missing_ingredients=policy_spec.max_missing_ingredients,
            rescue_minimum_coverage=policy_spec.rescue_minimum_coverage,
            rescue_max_missing_ingredients=policy_spec.rescue_max_missing_ingredients,
        ),
    )
    rows: list[dict[str, object]] = []
    for case in cases:
        candidate_ids = retriever.retrieve(case.request, limit=TOP_K)
        candidates = repository.get_many(candidate_ids)
        details = [
            ingredient_match_details(
                recipe,
                case.request.available_ingredients,
                rules.pantry_staples,
                rules.ingredient_aliases,
            )
            for recipe in candidates
        ]
        utilizations = [detail.input_utilization for detail in details]
        recipe_coverages = [detail.recipe_coverage for detail in details]
        missing_counts = [detail.missing_count for detail in details]
        best_utilization = max(utilizations, default=None)
        low_relevance_count = sum(
            utilization < LOW_RELEVANCE_CUTOFF for utilization in utilizations
        )
        unconstrained_ids = retriever.retrieve(
            _unconstrained_case(case).request,
            limit=TOP_K,
        )
        actual_corpus_no_result = not candidate_ids
        expected_corpus_no_result = case.expected_corpus_no_result
        label_matches = (
            expected_corpus_no_result is None
            or actual_corpus_no_result == expected_corpus_no_result
        )
        rows.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "category": (case.notes or "").split(";", maxsplit=1)[0],
                "policy": policy_spec.label,
                "minimum_coverage": policy_spec.minimum_coverage,
                "max_missing_ingredients": policy_spec.max_missing_ingredients,
                "expected_no_result": case.expected_no_result,
                "expected_corpus_no_result": expected_corpus_no_result,
                "actual_corpus_no_result": actual_corpus_no_result,
                "corpus_no_result_label_matches": label_matches,
                "candidate_count": len(candidate_ids),
                "best_input_utilization": best_utilization,
                "full_input_hit_at_10": best_utilization == 1.0,
                "low_relevance_candidate_count": low_relevance_count,
                "below_025_recipe_coverage_count": sum(
                    coverage < 0.25 for coverage in recipe_coverages
                ),
                "above_6_missing_ingredient_count": sum(
                    missing_count > 6 for missing_count in missing_counts
                ),
                "top10_removed_by_constraints": len(
                    set(unconstrained_ids) - set(candidate_ids)
                ),
                "candidate_recipe_ids_json": json.dumps(candidate_ids),
                "candidate_titles_json": json.dumps(
                    [recipe.title for recipe in candidates], ensure_ascii=False
                ),
                "candidate_input_utilizations_json": json.dumps(utilizations),
                "candidate_recipe_coverages_json": json.dumps(recipe_coverages),
                "candidate_missing_counts_json": json.dumps(missing_counts),
            }
        )
    return rows


def _summary_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    keys = sorted({(str(row["split"]), str(row["policy"])) for row in rows})
    for split, policy in keys:
        selected = [
            row
            for row in rows
            if row["split"] == split and row["policy"] == policy
        ]
        expected_result = [
            row for row in selected if row["expected_corpus_no_result"] is False
        ]
        candidate_total = sum(int(row["candidate_count"]) for row in selected)
        low_total = sum(int(row["low_relevance_candidate_count"]) for row in selected)
        low_coverage_total = sum(
            int(row["below_025_recipe_coverage_count"]) for row in selected
        )
        high_burden_total = sum(
            int(row["above_6_missing_ingredient_count"]) for row in selected
        )
        best_values = [
            float(row["best_input_utilization"])
            for row in selected
            if row["best_input_utilization"] is not None
        ]
        summaries.append(
            {
                "split": split,
                "policy": policy,
                "case_count": len(selected),
                "candidate_no_result_case_count": sum(
                    bool(row["actual_corpus_no_result"]) for row in selected
                ),
                "false_no_result_case_count": sum(
                    bool(row["actual_corpus_no_result"])
                    and row["expected_corpus_no_result"] is False
                    for row in selected
                ),
                "unexpected_result_case_count": sum(
                    not bool(row["actual_corpus_no_result"])
                    and row["expected_corpus_no_result"] is True
                    for row in selected
                ),
                "corpus_no_result_label_match_rate": sum(
                    bool(row["corpus_no_result_label_matches"]) for row in selected
                )
                / len(selected),
                "full_input_hit_case_count": sum(
                    bool(row["full_input_hit_at_10"]) for row in expected_result
                ),
                "expected_result_case_count": len(expected_result),
                "full_input_hit_rate_expected_result_cases": (
                    sum(bool(row["full_input_hit_at_10"]) for row in expected_result)
                    / len(expected_result)
                    if expected_result
                    else None
                ),
                "mean_best_input_utilization_nonempty": (
                    fmean(best_values) if best_values else None
                ),
                "candidate_count": candidate_total,
                "low_relevance_candidate_count": low_total,
                "low_relevance_candidate_share": (
                    low_total / candidate_total if candidate_total else None
                ),
                "below_025_recipe_coverage_count": low_coverage_total,
                "below_025_recipe_coverage_share": (
                    low_coverage_total / candidate_total if candidate_total else None
                ),
                "above_6_missing_ingredient_count": high_burden_total,
                "above_6_missing_ingredient_share": (
                    high_burden_total / candidate_total if candidate_total else None
                ),
            }
        )
    return summaries


def _equivalence_checks(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Compare the English and raw-Chinese forms of the same request."""

    checks: list[dict[str, object]] = []
    for policy in (spec.label for spec in POLICIES):
        english = next(
            row for row in rows if row["case_id"] == "D01" and row["policy"] == policy
        )
        chinese = next(
            row for row in rows if row["case_id"] == "D02" and row["policy"] == policy
        )
        checks.append(
            {
                "policy": policy,
                "ranked_candidate_ids_identical": (
                    english["candidate_recipe_ids_json"]
                    == chinese["candidate_recipe_ids_json"]
                ),
                "english_best_input_utilization": english["best_input_utilization"],
                "chinese_best_input_utilization": chinese["best_input_utilization"],
            }
        )
    return checks


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _markdown_report(
    summaries: Sequence[dict[str, object]],
    rows: Sequence[dict[str, object]],
    equivalence_checks: Sequence[dict[str, object]],
) -> str:
    lines = [
        "# V3 retrieval policy scan",
        "",
        (
            "Low relevance is defined before inspection as a Top-10 candidate using "
            "less than 50% of the user's non-pantry input ingredients. The last "
            "two columns expose low recipe coverage and extra shopping burden."
        ),
        "",
        (
            "| Split | Policy | No result | False no result | Full-input hit | "
            "Low input relevance | Coverage <.25 | Missing >6 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        full_rate = summary["full_input_hit_rate_expected_result_cases"]
        low_share = summary["low_relevance_candidate_share"]
        lines.append(
            "| {split} | {policy} | {no_result}/{cases} | {false_no_result} | "
            "{full_hits}/{expected} ({full_rate}) | {low_count}/{candidate_count} "
            "({low_share}) | {low_coverage}/{candidate_count} | "
            "{high_burden}/{candidate_count} |".format(
                split=summary["split"],
                policy=summary["policy"],
                no_result=summary["candidate_no_result_case_count"],
                cases=summary["case_count"],
                false_no_result=summary["false_no_result_case_count"],
                full_hits=summary["full_input_hit_case_count"],
                expected=summary["expected_result_case_count"],
                full_rate=(f"{float(full_rate):.3f}" if full_rate is not None else "n/a"),
                low_count=summary["low_relevance_candidate_count"],
                candidate_count=summary["candidate_count"],
                low_share=(f"{float(low_share):.3f}" if low_share is not None else "n/a"),
                low_coverage=summary["below_025_recipe_coverage_count"],
                high_burden=summary["above_6_missing_ingredient_count"],
            )
        )

    lines.extend(["", "## Chinese and English equivalence", ""])
    for check in equivalence_checks:
        lines.append(
            f"- D01 vs D02 / {check['policy']}: ranked candidate IDs identical = "
            f"{check['ranked_candidate_ids_identical']}; English best = "
            f"{check['english_best_input_utilization']}; Chinese best = "
            f"{check['chinese_best_input_utilization']}."
        )

    lines.extend(["", "## Focus cases", ""])
    focus_ids = {
        "D01",
        "D02",
        "D04",
        "D05",
        "D06",
        "D07",
        "D10",
        "D16",
        "F12",
        "F18",
        "F19",
    }
    for row in rows:
        if row["case_id"] not in focus_ids:
            continue
        best = row["best_input_utilization"]
        best_text = "none" if best is None else f"{float(best):.3f}"
        lines.append(
            f"- {row['case_id']} / {row['policy']}: candidates={row['candidate_count']}, "
            f"best_input_utilization={best_text}, "
            f"full_input_hit={row['full_input_hit_at_10']}, "
            f"low_relevance={row['low_relevance_candidate_count']}."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    dev_cases = load_cases_csv(args.dev_cases)
    final_cases = load_cases_csv(args.final_cases)
    if len(final_cases) != 20:
        raise ValueError(f"Expected exactly 20 final v3 cases; found {len(final_cases)}")
    _validate_v3_cases([*dev_cases, *final_cases])

    repository = ParquetRecipeRepository()
    rules = RetrievalRules.from_yaml()
    rows: list[dict[str, object]] = []
    for policy_spec in POLICIES:
        rows.extend(
            _case_rows(
                [*dev_cases, *final_cases],
                repository=repository,
                rules=rules,
                policy_spec=policy_spec,
            )
        )

    summaries = _summary_rows(rows)
    equivalence_checks = _equivalence_checks(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "case_policy_scan.csv", rows)
    _write_csv(args.output_dir / "policy_summary.csv", summaries)
    (args.output_dir / "policy_scan.json").write_text(
        json.dumps(
            {
                "definition": {
                    "top_k": TOP_K,
                    "low_relevance_input_utilization_below": LOW_RELEVANCE_CUTOFF,
                },
                "hashes": {
                    "corpus_sha256": _sha256(DEFAULT_CURATED_CORPUS_PATH),
                    "dev_cases_sha256": _sha256(args.dev_cases),
                    "final_cases_sha256": _sha256(args.final_cases),
                },
                "summaries": summaries,
                "equivalence_checks": equivalence_checks,
                "cases": rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "POLICY_SCAN.md").write_text(
        _markdown_report(summaries, rows, equivalence_checks), encoding="utf-8"
    )
    print(f"Saved offline policy scan: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
