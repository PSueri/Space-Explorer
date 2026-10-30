import asyncio

import httpx2

from backend.app.core.settings import Settings
from backend.app.integrations.ollama import OllamaClient


def test_adapter_keeps_original_text_and_generation_metadata() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "model": "test-model",
                "message": {
                    "role": "assistant",
                    "content": "  Original answer [1].\n",
                },
                "done": True,
                "done_reason": "length",
                "prompt_eval_count": 50,
                "eval_count": 128,
            },
        )

    async def run():
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler),
            base_url="http://localhost:11434",
        ) as http_client:
            adapter = OllamaClient(
                http_client,
                Settings(_env_file=None, llm_model="test-model"),
            )
            detailed = await adapter.generate_with_details("System", "Question")
            plain = await adapter.generate("System", "Question")
            return detailed, plain

    detailed, plain = asyncio.run(run())

    assert detailed.answer == "  Original answer [1].\n"
    assert plain == "Original answer [1]."
    assert detailed.model == "test-model"
    assert detailed.done_reason == "length"
    assert detailed.prompt_eval_count == 50
    assert detailed.eval_count == 128


def test_missing_provider_metadata_remains_unknown() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "Answer."},
                "done": True,
            },
        )

    async def run():
        async with httpx2.AsyncClient(
            transport=httpx2.MockTransport(handler),
            base_url="http://localhost:11434",
        ) as http_client:
            adapter = OllamaClient(http_client, Settings(_env_file=None))
            return await adapter.generate_with_details("System", "Question")

    result = asyncio.run(run())
    assert result.done_reason is None
    assert result.model is None
    assert result.eval_count is None
