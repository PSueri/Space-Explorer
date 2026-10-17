from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.dependencies import get_chat_service
from backend.app.core.exceptions import (
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from backend.app.models.conversation import ConversationTurn
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.chat import ChatService

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    service: Annotated[ChatService, Depends(get_chat_service)],
) -> ChatResponse:
    try:
        history = [
            ConversationTurn(
                question=turn.question,
                answer=turn.answer,
            )
            for turn in payload.history
        ]

        answer = await service.answer(payload.message, history)
    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="The model took too long to respond.",
        ) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="The model provider is unavailable. Check its configuration.",
        ) from exc
    except LLMResponseError as exc:
        raise HTTPException(
            status_code=502,
            detail="The model provider returned an invalid response.",
        ) from exc

    return ChatResponse(answer=answer)
