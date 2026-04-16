from google import genai
from google.genai import types
from src.backend.config import get_settings
from src.backend.prompts import SYSTEM_INSTRUCTION, build_context_prompt

settings = get_settings()


class GeminiClient:
    """
    Wraps Google's Gemini API for WorkMate.
    Two methods:
        ask_workmate()        — sync, returns full string response
        ask_workmate_stream() — async generator, yields tokens as they arrive
    Both use temperature=0.2 for factual, grounded responses.
    """

    MODEL = "gemini-2.0-flash"

    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.config = types.GenerateContentConfig(
            temperature=0.2,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        print("GeminiClient initialised ✓")

    def ask_workmate(self, context: str, question: str) -> str:
        """
        Synchronous generation.
        Sends context + question, returns the full response as a string.
        Used by the sync message endpoint in Step 2.6.
        """
        prompt = build_context_prompt(context, question)

        response = self.client.models.generate_content(
            model=self.MODEL,
            contents=prompt,
            config=self.config,
        )
        return response.text

    async def ask_workmate_stream(self, context: str, question: str):
        """
        Async streaming generator.
        Yields text tokens one at a time as Gemini generates them.
        Used by the SSE streaming endpoint in Step 2.6.

        Usage:
            async for token in client.ask_workmate_stream(context, question):
                yield token
        """
        prompt = build_context_prompt(context, question)

        async for chunk in await self.client.aio.models.generate_content_stream(
            model=self.MODEL,
            contents=prompt,
            config=self.config,
        ):
            if chunk.text:
                yield chunk.text