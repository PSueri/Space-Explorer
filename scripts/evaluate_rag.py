import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx2

from backend.app.core.settings import PROJECT_ROOT
from backend.app.schemas.rag import AskSourcesResponse
from evaluations.checks import (
    EvaluationCase,
    EvaluationDataset,
    evaluate_response,
)

DEFAULT_DATASET = PROJECT_ROOT / "evaluations" / "cassini_rag.json"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "evaluation-reports"


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

        result["response"] = payload.model_dump(mode="json")
        result["checks"] = checks
        result["automatic_checks_passed"] = all(checks.values())
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
        description="Evaluate Ask the Sources against a curated question dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
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

                print(f"{label}: {case.case_id} ({result['duration_seconds']:.3f}s)")

        passed = sum(result["automatic_checks_passed"] is True for result in results)

        report = {
            "dataset": dataset.model_dump(mode="json"),
            "api_base_url": arguments.base_url,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "summary": {
                "total_cases": len(results),
                "automatic_checks_passed": passed,
                "automatic_checks_failed": len(results) - passed,
                "manual_reviews_pending": len(results),
            },
            "results": results,
        }

        destination = save_report(report, arguments.output_directory)

        print(f"\nAutomatic checks passed: {passed}/{len(results)}")
        print(f"Report: {destination}")
        print("Manual review is required to assess factual correctness and support.")

        return 0 if passed == len(results) else 1
    except (httpx2.HTTPError, ValueError, OSError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
