from pydantic import BaseModel, ConfigDict, Field, field_validator


class HistoryTurn(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    message: str = Field(min_length=1, max_length=2000)
    history: list[HistoryTurn] = Field(default_factory=list, max_length=2)

    @field_validator("history")
    @classmethod
    def validate_history_size(
        cls,
        history: list[HistoryTurn],
    ) -> list[HistoryTurn]:
        total_characters = sum(
            len(turn.question) + len(turn.answer) for turn in history
        )

        if total_characters > 2000:
            raise ValueError("Conversation history exceeds 2000 characters.")

        return history


class ChatResponse(BaseModel):
    answer: str
