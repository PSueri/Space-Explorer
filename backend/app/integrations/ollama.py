import asyncio
from collections.abc import Sequence
from typing import Literal

import httpx2
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from backend.app.core.exceptions import (
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from backend.app.core.settings import Settings
from backend.app.models.conversation import ConversationTurn


class _OllamaMessage(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    role: Literal["assistant"]
    content: str = Field(min_length=1)


class _OllamaResponse(BaseModel):
    message: _OllamaMessage
    done: Literal[True]


class OllamaClient:
    def __init__(
        self,
        http_client: httpx2.AsyncClient,
        settings: Settings,
    ) -> None:
        self._http_client = http_client
        self._settings = settings

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        history: Sequence[ConversationTurn] = (),
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        for turn in history:
            messages.extend(
                [
                    {"role": "user", "content": turn.question},
                    {"role": "assistant", "content": turn.answer},
                ]
            )

        messages.append({"role": "user", "content": user_message})

        request_body = {
            "model": self._settings.llm_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "num_ctx": self._settings.llm_context_size,
                "num_predict": self._settings.llm_max_output_tokens,
            },
        }

        try:
            async with asyncio.timeout(self._settings.llm_timeout_seconds):
                response = await self._http_client.post(
                    "/api/chat",
                    json=request_body,
                )
                response.raise_for_status()
                payload = _OllamaResponse.model_validate(response.json())
        except (TimeoutError, httpx2.TimeoutException) as exc:
            raise LLMTimeoutError("Model request timed out.") from exc
        except httpx2.HTTPError as exc:
            raise LLMUnavailableError("Model provider is unavailable.") from exc
        except (ValueError, ValidationError) as exc:
            raise LLMResponseError("Model provider returned invalid data.") from exc

        return payload.message.content
