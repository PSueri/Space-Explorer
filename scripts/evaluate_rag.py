import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

import httpx2

from backend.app.core.settings import PROJECT_ROOT
from backend.app.schemas.rag import AskSourcesResponse
from evaluations.checks import (
    EvaluationCase,
    EvaluationDataset,
    evaluate_response,
    matches_all,
)

DEFAULT_DATASET = PROJECT_ROOT / "evaluations" / "cassini_rag.json"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "evaluation-reports"


def client_environment() -> dict:
    packages = {}

    for package in (
        "fastapi",
        "httpx2",
        "pydantic",
        "pydantic-settings",
        "qdrant-client",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None

    return {
        "python": platform.python_version(),
        "packages": packages,
    }


def run_case(
    client: httpx2.Client,
    case: EvaluationCase,
) -> dict:
    started = perf_counter()

    result = {
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "expected_status": case.expected_status,
        "reference_answer": case.reference_answer,
        "http_status": None,
        "response": None,
        "diagnostics": None,
        "retrieval_checks": {
            "expected_evidence_patterns_in_retrieved_passages": None,
        },
        "checks": {},
        "automatic_checks_passed": False,
        "error": None,
        "manual_review": {
            "status": "pending",
            "answer_correct": None,
            "claims_supported": None,
            "notes": "",
        },
    }

    try:
        response = client.post(
            "/ask-sources",
            json={"question": case.question},
        )
        result["http_status"] = response.status_code
        response.raise_for_status()

        payload = AskSourcesResponse.model_validate(response.json())
        checks = evaluate_response(case, payload)

        result["response"] = payload.model_dump(
            mode="json",
            exclude={"diagnostics"},
        )
        result["checks"] = checks
        result["automatic_checks_passed"] = all(checks.values())

        if payload.diagnostics is not None:
            result["diagnostics"] = payload.diagnostics.model_dump(mode="json")

            if case.evidence_patterns:
                retrieved_text = "\n\n".join(
                    item.record.chunk.text
                    for item in payload.diagnostics.retrieval.results
                )
                result["retrieval_checks"][
                    "expected_evidence_patterns_in_retrieved_passages"
                ] = matches_all(case.evidence_patterns, retrieved_text)
    except httpx2.TimeoutException:
        result["error"] = "The API request timed out."
    except httpx2.HTTPStatusError as exc:
        result["error"] = f"The API returned HTTP {exc.response.status_code}."
    except httpx2.HTTPError:
        result["error"] = "The API request could not be completed."
    except ValueError as exc:
        result["error"] = f"The API response was invalid: {exc}"

    result["duration_seconds"] = round(perf_counter() - started, 3)
    return result


def save_report(report: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = directory / f"rag-evaluation-{timestamp}.json"
    temporary_path = destination.with_suffix(".json.tmp")

    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(destination)

    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Ask the Sources and collect optional RAG diagnostics."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    arguments = parser.parse_args()

    try:
        dataset = EvaluationDataset.model_validate_json(
            arguments.dataset.read_text(encoding="utf-8")
        )
        started_at = datetime.now(UTC)

        with httpx2.Client(
            base_url=arguments.base_url,
            timeout=httpx2.Timeout(150.0, connect=5.0),
            trust_env=False,
        ) as client:
            health = client.get("/health", timeout=5.0)
            health.raise_for_status()

            results = []

            for case in dataset.cases:
                result = run_case(client, case)
                results.append(result)

                label = (
                    "CHECKS PASS"
                    if result["automatic_checks_passed"]
                    else "CHECKS FAIL"
                )
                diagnostics_label = (
                    "captured" if result["diagnostics"] is not None else "unavailable"
                )

                print(
                    f"{label}: {case.case_id} "
                    f"({result['duration_seconds']:.3f}s, "
                    f"diagnostics: {diagnostics_label})"
                )

        passed = sum(result["automatic_checks_passed"] is True for result in results)
        captured = sum(result["diagnostics"] is not None for result in results)

        report = {
            "report_version": "2",
            "dataset": dataset.model_dump(mode="json"),
            "api_base_url": arguments.base_url,
            "evaluation_client_environment": client_environment(),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "summary": {
                "total_cases": len(results),
                "automatic_checks_passed": passed,
                "automatic_checks_failed": len(results) - passed,
                "diagnostics_captured": captured,
                "manual_reviews_pending": len(results),
            },
            "results": results,
        }

        destination = save_report(report, arguments.output_directory)

        print(f"\nAutomatic checks passed: {passed}/{len(results)}")
        print(f"Diagnostics captured: {captured}/{len(results)}")
        print(f"Report: {destination}")
        print("Manual review is required to assess factual correctness and support.")

        if captured < len(results):
            print(
                "Some requests have no diagnostics. For successful API calls, "
                "enable APP_RAG_DIAGNOSTICS_ENABLED and restart the backend."
            )

        return 0 if passed == len(results) else 1
    except (httpx2.HTTPError, ValueError, OSError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
