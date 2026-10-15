from collections.abc import AsyncIterator
from typing import Annotated

import httpx2
from fastapi import Depends, Request

from backend.app.core.settings import Settings
from backend.app.integrations.ollama import OllamaClient
from backend.app.services.chat import ChatService
from backend.app.services.llm import LLMClient


async def get_llm_client(request: Request) -> AsyncIterator[LLMClient]:
    settings: Settings = request.app.state.settings

    async with httpx2.AsyncClient(
        base_url=str(settings.llm_base_url),
        timeout=settings.llm_timeout_seconds,
        trust_env=False,
    ) as http_client:
        yield OllamaClient(http_client, settings)


def get_chat_service(
    llm_client: Annotated[LLMClient, Depends(get_llm_client)],
) -> ChatService:
    return ChatService(llm_client)
