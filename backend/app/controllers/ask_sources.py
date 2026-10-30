from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.dependencies import get_rag_service
from backend.app.core.exceptions import (
    LLMResponseError,
    LLMTimeoutError,
    LLMUnavailableError,
    RetrievalUnavailableError,
)
from backend.app.schemas.rag import AskSourcesRequest, AskSourcesResponse
from backend.app.services.rag import RagService

router = APIRouter(tags=["sources"])


@router.post("/ask-sources", response_model=AskSourcesResponse)
async def ask_sources(
    payload: AskSourcesRequest,
    service: Annotated[RagService, Depends(get_rag_service)],
) -> AskSourcesResponse:
    try:
        result = await service.answer(payload.question)
    except RetrievalUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="Document search is unavailable. Check the index and Ollama.",
        ) from exc
    except LLMTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="The model took too long to respond.",
        ) from exc
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail="The model provider is unavailable.",
        ) from exc
    except LLMResponseError as exc:
        raise HTTPException(
            status_code=502,
            detail="The model provider returned an invalid response.",
        ) from exc

    return AskSourcesResponse.model_validate(asdict(result))
