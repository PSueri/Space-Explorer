import pytest
from fastapi.testclient import TestClient

from backend.app.core.dependencies import get_llm_client
from backend.app.core.exceptions import (
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from backend.app.core.settings import Settings
from backend.app.main import create_app


class FakeLLMClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.messages: list[str] = []

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
    ) -> str:
        self.messages.append(user_message)

        if self.error is not None:
            raise self.error

        return "An orbiter circles a celestial body; a lander reaches its surface."


def make_application(fake: FakeLLMClient):
    settings = Settings(
        _env_file=None,
        name="Space Explorer Test",
        debug=False,
    )
    application = create_app(settings)
    application.dependency_overrides[get_llm_client] = lambda: fake
    return application


def test_chat_returns_answer() -> None:
    fake = FakeLLMClient()

    with TestClient(make_application(fake)) as client:
        response = client.post(
            "/chat",
            json={"message": "  What is an orbiter?  "},
        )

    assert response.status_code == 200
    assert "orbiter" in response.json()["answer"]
    assert fake.messages == ["What is an orbiter?"]


@pytest.mark.parametrize("message", ["   ", "x" * 2001])
def test_chat_rejects_invalid_input(message: str) -> None:
    fake = FakeLLMClient()

    with TestClient(make_application(fake)) as client:
        response = client.post("/chat", json={"message": message})

    assert response.status_code == 422
    assert fake.messages == []


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (LLMTimeoutError(), 504),
        (LLMUnavailableError(), 503),
        (LLMResponseError(), 502),
    ],
)
def test_chat_handles_provider_failure(
    error: Exception,
    expected_status: int,
) -> None:
    fake = FakeLLMClient(error)

    with TestClient(make_application(fake)) as client:
        response = client.post(
            "/chat",
            json={"message": "What is an orbiter?"},
        )

    assert response.status_code == expected_status
    assert "detail" in response.json()
