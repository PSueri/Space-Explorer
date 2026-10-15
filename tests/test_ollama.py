import asyncio

import httpx2
import pytest

from backend.app.core.exceptions import (
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from backend.app.core.settings import Settings
from backend.app.integrations.ollama import OllamaClient


async def generate_with_transport(
    transport: httpx2.MockTransport,
) -> str:
    settings = Settings(
        _env_file=None,
        llm_model="test-model",
        llm_timeout_seconds=5,
        llm_context_size=2048,
        llm_max_output_tokens=128,
    )

    async with httpx2.AsyncClient(
        transport=transport,
        base_url="http://localhost:11434",
    ) as http_client:
        adapter = OllamaClient(http_client, settings)
        return await adapter.generate("Answer in English.", "What is an orbiter?")


@pytest.mark.parametrize(
    ("status_code", "response_body", "expected_error"),
    [
        (503, {}, LLMUnavailableError),
        (200, {"message": {"content": ""}, "done": True}, LLMResponseError),
    ],
)
def test_adapter_rejects_failed_responses(
    status_code: int,
    response_body: dict,
    expected_error: type[Exception],
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status_code, json=response_body)

    with pytest.raises(expected_error):
        asyncio.run(generate_with_transport(httpx2.MockTransport(handler)))


def test_adapter_handles_timeout() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("Simulated timeout.", request=request)

    with pytest.raises(LLMTimeoutError):
        asyncio.run(generate_with_transport(httpx2.MockTransport(handler)))


def test_adapter_returns_valid_content() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/chat"

        return httpx2.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "An orbiter circles a celestial body.",
                },
                "done": True,
            },
        )

    answer = asyncio.run(generate_with_transport(httpx2.MockTransport(handler)))

    assert answer == "An orbiter circles a celestial body."
