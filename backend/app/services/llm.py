from collections.abc import Sequence
from typing import Protocol

from backend.app.models.conversation import ConversationTurn


class LLMClient(Protocol):
    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        history: Sequence[ConversationTurn] = (),
    ) -> str: ...
