import pytest
from pydantic import ValidationError

from backend.app.schemas.rag import AskSourcesResponse
from evaluations.checks import EvaluationCase, evaluate_response


def make_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="example-landing",
        category="answerable",
        question="Where did the fictional probe land?",
        expected_status="answered",
        reference_answer="The probe landed on Example Moon.",
        answer_patterns=[r"\bExample Moon\b"],
        evidence_patterns=[r"landed on Example Moon"],
        expected_source_urls=["https://example.com/mission/"],
    )


def make_response(
    *,
    url: str = "https://example.com/mission/",
    excerpt: str = "The fictional probe landed on Example Moon.",
) -> AskSourcesResponse:
    return AskSourcesResponse.model_validate(
        {
            "status": "answered",
            "answer": "The probe landed on Example Moon [1].",
            "citations": [
                {
                    "number": 1,
                    "chunk_id": "example-chunk",
                    "title": "Example mission",
                    "url": url,
                    "excerpt": excerpt,
                }
            ],
        }
    )


def test_supported_example_passes_automatic_checks() -> None:
    checks = evaluate_response(make_case(), make_response())

    assert all(checks.values())


def test_unexpected_source_is_detected() -> None:
    response = make_response(url="https://other.example/mission/")

    checks = evaluate_response(make_case(), response)

    assert checks["expected_sources"] is False


def test_irrelevant_cited_evidence_is_detected() -> None:
    response = make_response(
        excerpt="The fictional mission studied atmospheric temperatures."
    )

    checks = evaluate_response(make_case(), response)

    assert checks["evidence_patterns"] is False


def test_abstention_on_answerable_question_is_detected() -> None:
    response = AskSourcesResponse(
        status="insufficient_evidence",
        answer="Insufficient evidence.",
        citations=[],
    )

    checks = evaluate_response(make_case(), response)

    assert checks["expected_status"] is False
    assert not all(checks.values())


def test_answer_on_unanswerable_question_is_detected() -> None:
    case = EvaluationCase(
        case_id="missing-cost",
        category="unanswerable",
        question="What was the mission cost?",
        expected_status="insufficient_evidence",
        reference_answer="The document does not provide the cost.",
    )

    checks = evaluate_response(case, make_response())

    assert checks["expected_status"] is False
    assert checks["no_citations"] is False


def test_invalid_regex_is_rejected() -> None:
    data = make_case().model_dump(mode="json")
    data["answer_patterns"] = ["["]

    with pytest.raises(ValidationError):
        EvaluationCase.model_validate(data)
