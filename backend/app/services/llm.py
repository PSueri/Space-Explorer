from collections.abc import Sequence
from typing import Protocol

from backend.app.models.conversation import ConversationTurn
from backend.app.models.generation import GenerationResult


class LLMClient(Protocol):
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        history: Sequence[ConversationTurn] = (),
    ) -> str: ...

    async def generate_with_details(
        self,
        system_prompt: str,
        user_message: str,
        history: Sequence[ConversationTurn] = (),
    ) -> GenerationResult: ...
