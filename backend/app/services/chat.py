from backend.app.services.llm import LLMClient

SYSTEM_PROMPT = (
    "You are Space Explorer, an educational assistant about space exploration. "
    "Answer in English using clear, concise language. "
    "Focus on astronomy, spacecraft, and space missions. "
    "If a question is unrelated, briefly explain your scope. "
    "When uncertain, say so. Do not invent sources or citations. "
    "Keep answers short."
)


class ChatService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    async def answer(self, message: str) -> str:
        return await self._llm_client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_message=message,
        )
