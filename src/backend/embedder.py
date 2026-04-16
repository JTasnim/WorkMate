import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types
from src.backend.config import get_settings


EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 100
BATCH_DELAY = 0.7


class GoogleEmbedder:
    def __init__(self):
        api_key = get_settings().GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        self.client = genai.Client(api_key=api_key)

    def embed_text(self, text: str) -> list[float]:
        """Embed a single document passage for storage."""
        result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return result.embeddings[0].values

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query for retrieval."""
        result = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts in batches to respect API limits."""
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            for text in batch:
                embeddings.append(self.embed_text(text))
                time.sleep(BATCH_DELAY)
        return embeddings