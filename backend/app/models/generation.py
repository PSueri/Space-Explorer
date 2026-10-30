from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model: str | None = None
    done_reason: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
