import httpx2

from evaluations.checks import EvaluationCase
from scripts.evaluate_rag import run_case


def make_case() -> EvaluationCase:
    return EvaluationCase(
        case_id="example-evidence",
        category="answerable",
        question="Where did the example probe land?",
        expected_status="answered",
        reference_answer="The probe landed on Example Moon.",
        answer_patterns=["Example Moon"],
        evidence_patterns=["landed on Example Moon"],
        expected_source_urls=["https://example.com/mission/"],
    )


def make_trace() -> dict:
    return {
        "configuration": {
            "provider": "ollama",
            "requested_model": "test-model",
            "context_size": 2048,
            "max_output_tokens": 128,
        },
        "retrieval": {
            "duration_seconds": 0.1,
            "requested_limit": 2,
            "results": [
                {
                    "score": 0.8,
                    "record": {
                        "chunk": {
                            "chunk_id": "example-chunk",
                            "index": 0,
                            "start": 0,
                            "end": 33,
                            "text": "The probe landed on Example Moon.",
                        },
                        "source": {
                            "source_id": "example-mission",
                            "title": "Example mission",
                            "publisher": "Example publisher",
                            "url": "https://example.com/mission/",
                            "usage_guidelines_url": "https://example.com/usage/",
                        },
                        "fetched_url": "https://example.com/mission/",
                        "retrieved_at": "2026-01-01T00:00:00Z",
                        "content_hash": "example-hash",
                        "parser_version": "2",
                        "chunker_version": "2",
                    },
                }
            ],
        },
        "generation": {
            "attempted": True,
            "raw_answer": "Unsupported answer without citations.",
            "done_reason": "stop",
            "output_truncated": False,
        },
        "abstention_reason": "missing_citations",
    }


def test_report_can_separate_retrieved_evidence_from_discarded_generation() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "status": "insufficient_evidence",
                "answer": "I could not produce an answer supported by the passages.",
                "citations": [],
                "diagnostics": make_trace(),
            },
        )

    with httpx2.Client(
        transport=httpx2.MockTransport(handler),
        base_url="http://testserver",
    ) as client:
        result = run_case(client, make_case())

    assert result["automatic_checks_passed"] is False
    assert result["checks"]["evidence_patterns"] is False
    assert (
        result["retrieval_checks"]["expected_evidence_patterns_in_retrieved_passages"]
        is True
    )
    assert result["diagnostics"]["abstention_reason"] == "missing_citations"
    assert result["diagnostics"]["generation"]["raw_answer"] == (
        "Unsupported answer without citations."
    )
    assert "diagnostics" not in result["response"]


def test_older_api_response_keeps_diagnostics_unavailable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "status": "insufficient_evidence",
                "answer": "Insufficient evidence.",
                "citations": [],
            },
        )

    with httpx2.Client(
        transport=httpx2.MockTransport(handler),
        base_url="http://testserver",
    ) as client:
        result = run_case(client, make_case())

    assert result["error"] is None
    assert result["diagnostics"] is None
    assert (
        result["retrieval_checks"]["expected_evidence_patterns_in_retrieved_passages"]
        is None
    )
