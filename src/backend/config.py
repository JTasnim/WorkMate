from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All application config read from .env at startup.
    Pydantic validates types and raises at startup if a required var is missing.
    Fields without defaults are required — the app will not start without them.
    Fields with defaults are optional.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # ignore unknown .env keys — won't raise on BATCH_SIZE etc.
    )

    # ── Notion ────────────────────────────────────────────────
    NOTION_TOKEN: str
    NOTION_API_VERSION: str = "2022-06-28"
    NOTION_DATA_PATH: str = "src/data/notion_data.json"

    # ── Google Gemini ─────────────────────────────────────────
    GEMINI_API_KEY: str

    # ── VoyageAI (optional — graceful degradation if missing) ─
    VOYAGE_API_KEY: str = ""

    # ── Google OAuth (Step 2.5) ───────────────────────────────
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    # ── JWT ───────────────────────────────────────────────────
    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_HOURS: int = 24

    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = "sqlite:///./workmate.db"

    # ── Paths ─────────────────────────────────────────────────
    CHROMA_PATH: str = "workmate_db/chroma"
    BM25_INDEX_PATH: str = "workmate_db/bm25_index.pkl"


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    lru_cache means .env is read once at startup, not on every request.
    Use: settings = get_settings()
    """
    return Settings()