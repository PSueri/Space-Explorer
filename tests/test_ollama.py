import asyncio
import json
from collections.abc import Sequence

import httpx2
import pytest

from backend.app.core.exceptions import (
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from backend.app.core.settings import Settings
from backend.app.integrations.ollama import OllamaClient
from backend.app.models.conversation import ConversationTurn


async def generate_with_transport(
    transport: httpx2.MockTransport,
    history: Sequence[ConversationTurn] = (),
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
        return await adapter.generate(
            "Answer in English.",
            "What is an orbiter?",
            history=history,
        )


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


def test_adapter_preserves_conversation_order() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)

        assert body["messages"] == [
            {"role": "system", "content": "Answer in English."},
            {"role": "user", "content": "What is a lander?"},
            {"role": "assistant", "content": "A spacecraft that lands."},
            {"role": "user", "content": "What is an orbiter?"},
        ]

        return httpx2.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "A spacecraft that orbits a celestial body.",
                },
                "done": True,
            },
        )

    history = [
        ConversationTurn(
            question="What is a lander?",
            answer="A spacecraft that lands.",
        )
    ]

    asyncio.run(
        generate_with_transport(
            httpx2.MockTransport(handler),
            history=history,
        )
    )
