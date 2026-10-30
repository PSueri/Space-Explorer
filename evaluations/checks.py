import re
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)

from backend.app.schemas.rag import AskSourcesResponse


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    category: Literal[
        "answerable",
        "unanswerable",
        "instruction_override",
    ]
    question: str = Field(min_length=1, max_length=500)
    expected_status: Literal["answered", "insufficient_evidence"]
    reference_answer: str = Field(min_length=1)
    answer_patterns: list[str] = Field(default_factory=list)
    evidence_patterns: list[str] = Field(default_factory=list)
    expected_source_urls: list[HttpUrl] = Field(default_factory=list)

    @field_validator("answer_patterns", "evidence_patterns")
    @classmethod
    def validate_patterns(cls, patterns: list[str]) -> list[str]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"Invalid regular expression: {pattern}") from exc

        return patterns

    @model_validator(mode="after")
    def validate_expectations(self) -> Self:
        if self.expected_status == "answered":
            if (
                not self.answer_patterns
                or not self.evidence_patterns
                or not self.expected_source_urls
            ):
                raise ValueError(
                    "Answerable cases require answer, evidence, and source checks."
                )

        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    description: str
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identifiers(self) -> Self:
        identifiers = [case.case_id for case in self.cases]

        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Evaluation case identifiers must be unique.")

        return self


def matches_all(patterns: list[str], text: str) -> bool:
    return all(
        re.search(pattern, text, flags=re.IGNORECASE) is not None
        for pattern in patterns
    )


def evaluate_response(
    case: EvaluationCase,
    response: AskSourcesResponse,
) -> dict[str, bool]:
    checks = {
        "expected_status": response.status == case.expected_status,
    }

    if case.expected_status == "insufficient_evidence":
        checks["no_citations"] = not response.citations
        return checks

    citation_numbers = [citation.number for citation in response.citations]
    referenced_numbers = {
        int(number) for number in re.findall(r"\[(\d+)\]", response.answer)
    }
    allowed_urls = {str(url) for url in case.expected_source_urls}
    evidence = "\n\n".join(citation.excerpt for citation in response.citations)

    checks.update(
        {
            "has_citations": bool(response.citations),
            "unique_citation_numbers": (
                len(citation_numbers) == len(set(citation_numbers))
            ),
            "valid_citation_numbers": all(number > 0 for number in citation_numbers),
            "references_match_citations": (
                bool(referenced_numbers) and referenced_numbers == set(citation_numbers)
            ),
            "expected_sources": (
                bool(response.citations)
                and all(
                    str(citation.url) in allowed_urls for citation in response.citations
                )
            ),
            "answer_patterns": matches_all(
                case.answer_patterns,
                response.answer,
            ),
            "evidence_patterns": matches_all(
                case.evidence_patterns,
                evidence,
            ),
        }
    )

    return checks
