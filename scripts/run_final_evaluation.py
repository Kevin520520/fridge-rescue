"""Run the frozen final Agentic-versus-single-prompt evaluation.

This entry point accepts the full typed request schema. The v3 frozen cases use
only constraints supported by the Epicurious-derived corpus; they intentionally
leave the unavailable source time field blank. Final cases require
``label_status=reviewed`` by default. Each case alternates the first evaluation
arm, and every repetition stores a separate report and append-only attempts log.

API keys are neither stored nor printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from fridge_rescue.baseline import ResponsesSinglePromptBaseline
from fridge_rescue.evaluation import (
    AgenticEvaluationRunner,
    CaseEvaluation,
    DirectBaselineEvaluationRunner,
    EvaluationCase,
    RunnerLike,
    load_cases_csv,
    run_interleaved_comparison,
    save_evaluation_results,
)
from fridge_rescue.openai_selector import ResponsesRecommendationSelector
from fridge_rescue.repository import DEFAULT_CURATED_CORPUS_PATH, ParquetRecipeRepository
from fridge_rescue.retrieval import DeterministicRecipeRetriever, RetrievalPolicy, RetrievalRules
from fridge_rescue.settings import PROJECT_ROOT, get_settings
from fridge_rescue.workflow import FridgeRescueWorkflow

DEFAULT_CASES_PATH = PROJECT_ROOT / "evaluation" / "final_cases_v3.csv"


class ProgressRunner:
    """Print a compact status line after every completed system-case run."""

    def __init__(self, runner: RunnerLike, *, label: str, total: int) -> None:
        self.runner = runner
        self.label = label
        self.total = total
        self.completed = 0

    def run(self, case: EvaluationCase):
        run_method = getattr(self.runner, "run", None)
        result = run_method(case) if callable(run_method) else self.runner(case)  # type: ignore[misc]
        self.completed += 1
        status = getattr(result, "status", "unknown")
        print(
            f"[{self.label} {self.completed:03d}/{self.total:03d}] "
            f"{case.case_id}: {status}",
            flush=True,
        )
        return result


class RecordingRetriever:
    """Expose the exact ordered candidates used by one workflow run."""

    def __init__(self, retriever: DeterministicRecipeRetriever) -> None:
        self.retriever = retriever
        self.last_retrieved_ids: list[str] | None = None

    def reset(self) -> None:
        self.last_retrieved_ids = None

    def retrieve(self, request, *, limit: int) -> list[str]:
        result = self.retriever.retrieve(request, limit=limit)
        self.last_retrieved_ids = list(result)
        return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative number")
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provider_label(provider: str) -> str:
    return {"deepseek": "DeepSeek", "openai": "OpenAI"}.get(provider, "LLM")


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = yaml.safe_load(source) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def _reviewed_substitution_notes(rules: Mapping[str, object]) -> tuple[str, ...]:
    notes: list[str] = []
    substitutions = rules.get("substitutions", {})
    if not isinstance(substitutions, Mapping):
        return ()
    for original, raw_rule in substitutions.items():
        if not isinstance(raw_rule, Mapping):
            continue
        substitute = str(raw_rule.get("substitute", "")).strip()
        warning = str(raw_rule.get("warning", "")).strip()
        contexts = raw_rule.get("allowed_contexts", [])
        if not substitute or not isinstance(contexts, list) or not contexts:
            continue
        context_text = ", ".join(str(item).strip() for item in contexts if str(item).strip())
        if not context_text:
            continue
        note = f"For {original}, {substitute} may work in: {context_text}."
        if warning:
            note = f"{note} {warning}"
        notes.append(note)
    return tuple(notes)


def _validate_final_cases(
    cases: Sequence[EvaluationCase],
    *,
    allow_placeholders: bool,
    expected_case_count: int,
) -> None:
    if not cases:
        raise ValueError("The final evaluation case file is empty")
    if len(cases) != expected_case_count:
        raise ValueError(
            f"Formal evaluation requires exactly {expected_case_count} cases; "
            f"found {len(cases)}"
        )
    case_ids = [case.case_id for case in cases]
    duplicate_ids = sorted(
        {case_id for index, case_id in enumerate(case_ids) if case_id in case_ids[:index]}
    )
    if duplicate_ids:
        raise ValueError("Formal evaluation case IDs must be unique: " + ", ".join(duplicate_ids))
    non_final = [case.case_id for case in cases if case.split != "final"]
    if non_final:
        raise ValueError("Final evaluation cases must use split=final: " + ", ".join(non_final))
    placeholders = [case.case_id for case in cases if case.label_status != "reviewed"]
    if placeholders and not allow_placeholders:
        raise ValueError(
            "Formal evaluation refuses placeholder cases. Review and label: "
            + ", ".join(placeholders)
        )


def _validate_development_cases(cases: Sequence[EvaluationCase]) -> None:
    """Keep development diagnostics separate from the frozen final split."""

    if not cases:
        raise ValueError("The development evaluation case file is empty")
    non_dev = [case.case_id for case in cases if case.split != "dev"]
    if non_dev:
        raise ValueError(
            "Development evaluation cases must use split=dev: " + ", ".join(non_dev)
        )


def _configured_final_case_count(config: Mapping[str, object]) -> int:
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise TypeError("config.yaml evaluation.final_cases must be a positive integer")
    value = evaluation.get("final_cases")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("config.yaml evaluation.final_cases must be a positive integer")
    return value


def _configured_development_case_count(config: Mapping[str, object]) -> int:
    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise TypeError("config.yaml evaluation.development_cases must be a positive integer")
    value = evaluation.get("development_cases")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            "config.yaml evaluation.development_cases must be a positive integer"
        )
    return value


def _validate_corpus_no_result_labels(
    cases: Sequence[EvaluationCase],
    retriever: DeterministicRecipeRetriever,
    *,
    top_k: int,
) -> None:
    """Refuse a run when frozen corpus labels no longer match this corpus."""

    mismatches: list[str] = []
    for case in cases:
        if case.expected_corpus_no_result is None:
            continue
        actual_no_result = not retriever.retrieve(case.request, limit=top_k)
        if actual_no_result != case.expected_corpus_no_result:
            mismatches.append(
                f"{case.case_id} (expected={case.expected_corpus_no_result}, "
                f"actual={actual_no_result})"
            )
    if mismatches:
        raise ValueError(
            "Corpus no-result labels are stale for the current corpus/policy: "
            + ", ".join(mismatches)
        )


def _source_paths() -> list[Path]:
    explicit = [
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "config.yaml",
        PROJECT_ROOT / "prompts.yaml",
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "requirements-lock.txt",
        PROJECT_ROOT / "rules.yaml",
    ]
    discovered = [
        *sorted((PROJECT_ROOT / "src").rglob("*.py")),
        *sorted((PROJECT_ROOT / "scripts").glob("*.py")),
        *sorted((PROJECT_ROOT / "tests").rglob("*.py")),
    ]
    return [path for path in dict.fromkeys([*explicit, *discovered]) if path.is_file()]


def build_source_manifest() -> dict[str, object]:
    """Return a secret-free manifest tying results to the exact source files."""

    files = [
        {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in _source_paths()
    ]
    canonical = json.dumps(files, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return {
        "algorithm": "sha256",
        "combined_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "files": files,
    }


def _attempt_logger(path: Path, repetition: int):
    """Return a callback that appends one durable JSONL row per system-case run."""

    sequence = 0

    def append_attempt(result: CaseEvaluation) -> None:
        nonlocal sequence
        sequence += 1
        row = {
            "sequence": sequence,
            "repetition": repetition,
            **result.model_dump(mode="json"),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    return append_attempt


def _build_runners(
    *,
    arms: str,
    repetitions: int,
    case_count: int,
    repository: ParquetRecipeRepository,
    retriever: DeterministicRecipeRetriever,
    rules: RetrievalRules,
    reviewed_notes: Sequence[str],
    top_k: int,
    input_cost_per_million_usd: float | None,
    output_cost_per_million_usd: float | None,
) -> dict[str, RunnerLike]:
    settings = get_settings()
    if not settings.has_valid_llm_key or settings.active_api_key is None:
        raise ValueError(f"Enter {settings.active_key_env_name} in .env first")

    api_key = settings.active_api_key.get_secret_value()
    label = _provider_label(settings.active_provider)
    reasoning_effort = "none" if settings.active_provider == "deepseek" else None
    total = repetitions * case_count
    runners: dict[str, RunnerLike] = {}

    if arms in {"both", "baseline"}:
        baseline = ResponsesSinglePromptBaseline(
            model=settings.active_model,
            api_key=api_key,
            base_url=settings.active_base_url,
            provider_name=label,
            reasoning_effort=reasoning_effort,
        )
        runners["baseline_a"] = ProgressRunner(
            DirectBaselineEvaluationRunner(
                baseline,
                input_cost_per_million_usd=input_cost_per_million_usd,
                output_cost_per_million_usd=output_cost_per_million_usd,
            ),
            label="baseline_a",
            total=total,
        )

    if arms in {"both", "agentic"}:
        selector = ResponsesRecommendationSelector(
            model=settings.active_model,
            api_key=api_key,
            base_url=settings.active_base_url,
            provider_name=label,
            reasoning_effort=reasoning_effort,
            allowed_substitution_notes=reviewed_notes,
            ingredient_aliases=rules.ingredient_aliases,
            pantry_staples=rules.pantry_staples,
        )
        recording_retriever = RecordingRetriever(retriever)
        workflow = FridgeRescueWorkflow(
            repository=repository,
            retriever=recording_retriever,
            selector=selector,
            top_k=top_k,
            allowed_substitution_notes=reviewed_notes,
            allergen_aliases=rules.allergen_aliases,
            ingredient_aliases=rules.ingredient_aliases,
            pantry_staples=rules.pantry_staples,
        )
        runners["agentic"] = ProgressRunner(
            AgenticEvaluationRunner(
                workflow,
                retrieval_diagnostics=recording_retriever,
                input_cost_per_million_usd=input_cost_per_million_usd,
                output_cost_per_million_usd=output_cost_per_million_usd,
            ),
            label="agentic",
            total=total,
        )
    return runners


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--repetitions", type=_positive_int, default=1)
    parser.add_argument("--arms", choices=("both", "baseline", "agentic"), default="both")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="Allow draft labels for a diagnostic run; never use this for reported final results.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate cases, corpus and configuration without making an API request.",
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Run explicitly marked split=dev cases without weakening formal-run safeguards.",
    )
    parser.add_argument(
        "--input-cost-per-million-usd",
        type=_non_negative_float,
        help="Current provider input-token price per one million tokens.",
    )
    parser.add_argument(
        "--output-cost-per-million-usd",
        type=_non_negative_float,
        help="Current provider output-token price per one million tokens.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if (args.input_cost_per_million_usd is None) != (
        args.output_cost_per_million_usd is None
    ):
        print(
            "[FAILED] Supply both token-price arguments or omit both; "
            "cost must not be partially estimated."
        )
        return 1
    if not args.cases.is_file():
        print(f"[FAILED] Case file not found: {args.cases}")
        return 1

    try:
        cases = load_cases_csv(args.cases)
        config = _load_yaml_mapping(PROJECT_ROOT / "config.yaml")
        expected_final_case_count = _configured_final_case_count(config)
        expected_development_case_count = _configured_development_case_count(config)
        if args.development:
            _validate_development_cases(cases)
            if len(cases) != expected_development_case_count:
                raise ValueError(
                    "Development evaluation requires exactly "
                    f"{expected_development_case_count} cases; found {len(cases)}"
                )
        else:
            _validate_final_cases(
                cases,
                allow_placeholders=args.allow_placeholder,
                expected_case_count=expected_final_case_count,
            )
        repository = ParquetRecipeRepository()
        rules = RetrievalRules.from_yaml()
        policy = RetrievalPolicy.from_yaml()
        retriever = DeterministicRecipeRetriever(repository, rules=rules, policy=policy)
        raw_rules = _load_yaml_mapping(PROJECT_ROOT / "rules.yaml")
        top_k = int(config["retrieval"]["candidate_pool_size"])
        _validate_corpus_no_result_labels(cases, retriever, top_k=top_k)
        reviewed_notes = _reviewed_substitution_notes(raw_rules)
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(f"[FAILED] {error}")
        return 1

    run_kind = "development" if args.development else "formal"
    print(
        f"[READY ] {len(cases)} {run_kind} cases; "
        f"{len(repository)} curated recipes; Top-{top_k}",
        flush=True,
    )
    if args.validate_only:
        print("Validation only: no API request was made.")
        return 0

    try:
        runners = _build_runners(
            arms=args.arms,
            repetitions=args.repetitions,
            case_count=len(cases),
            repository=repository,
            retriever=retriever,
            rules=rules,
            reviewed_notes=reviewed_notes,
            top_k=top_k,
            input_cost_per_million_usd=args.input_cost_per_million_usd,
            output_cost_per_million_usd=args.output_cost_per_million_usd,
        )
    except ValueError as error:
        print(f"[FAILED] {error}")
        return 1

    settings = get_settings()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        PROJECT_ROOT / "evaluation" / "results" / f"{run_kind}_{timestamp}"
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"[FAILED] Output directory already exists: {output_dir}")
        return 1

    shutil.copy2(args.cases, output_dir / "frozen_cases.csv")
    source_manifest = build_source_manifest()
    source_manifest_path = output_dir / "source_manifest.json"
    source_manifest_path.write_text(
        json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    base_metadata: dict[str, str | int | float | bool | None] = {
        "provider": settings.active_provider,
        "model": settings.active_model,
        "run_kind": run_kind,
        "arms": ",".join(runners),
        "repetitions": args.repetitions,
        "case_count": len(cases),
        "configured_final_case_count": expected_final_case_count,
        "configured_development_case_count": expected_development_case_count,
        "cases_sha256": _sha256(args.cases),
        "corpus_sha256": _sha256(DEFAULT_CURATED_CORPUS_PATH),
        "config_sha256": _sha256(PROJECT_ROOT / "config.yaml"),
        "rules_sha256": _sha256(PROJECT_ROOT / "rules.yaml"),
        "prompts_sha256": _sha256(PROJECT_ROOT / "prompts.yaml"),
        "source_manifest_sha256": str(source_manifest["combined_sha256"]),
        "started_at_utc": datetime.now(UTC).isoformat(),
        "input_cost_per_million_usd": args.input_cost_per_million_usd,
        "output_cost_per_million_usd": args.output_cost_per_million_usd,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(base_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for source_name in ("config.yaml", "rules.yaml", "prompts.yaml"):
        shutil.copy2(PROJECT_ROOT / source_name, output_dir / f"frozen_{source_name}")
    curation_summary = PROJECT_ROOT / "data" / "curated" / "curation_summary.json"
    if curation_summary.is_file():
        shutil.copy2(curation_summary, output_dir / "frozen_curation_summary.json")

    diet_markers = raw_rules.get("diet_conflict_markers", {})
    equipment_aliases = raw_rules.get("equipment_keywords", {})
    if not isinstance(diet_markers, dict):
        diet_markers = {}
    if not isinstance(equipment_aliases, dict):
        equipment_aliases = {}

    for repetition in range(1, args.repetitions + 1):
        repetition_dir = output_dir / f"repetition_{repetition:02d}"
        repetition_dir.mkdir()
        attempts_path = repetition_dir / "attempts.jsonl"

        metadata = {**base_metadata, "repetition": repetition}
        report = run_interleaved_comparison(
            cases,
            runners=runners,
            catalog=repository,
            allergen_aliases=rules.allergen_aliases,
            ingredient_aliases=rules.ingredient_aliases,
            diet_conflict_markers=diet_markers,
            equipment_aliases=equipment_aliases,
            pantry_staples=rules.pantry_staples,
            metadata=metadata,
            reverse_first_case_order=bool((repetition - 1) % 2),
            on_result=_attempt_logger(attempts_path, repetition),
        )
        save_evaluation_results(report, repetition_dir, stem="comparison")

        print(f"\nRepetition {repetition} summary", flush=True)
        for summary in report.summaries:
            print(
                f"- {summary.runner_name}: errors={summary.error_case_count}/"
                f"{summary.case_count}; success={summary.common_success_rate}; "
                "all-input-hit-applicable="
                f"{summary.applicable_all_requested_ingredients_case_rate} "
                f"(n={summary.all_requested_ingredients_applicable_case_count}); "
                f"all-input-hit-all-cases={summary.all_requested_ingredients_case_rate}; "
                f"mean_latency_ms={summary.mean_latency_ms:.1f}",
                flush=True,
            )

    completed_metadata = {
        **base_metadata,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "completed_repetitions": args.repetitions,
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(completed_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {run_kind} evaluation: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
