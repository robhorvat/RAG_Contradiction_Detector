from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    openai_api_key: str | None
    cohere_api_key: str | None
    gemini_api_key: str | None
    openai_chat_model: str
    gemini_model: str
    cohere_rerank_model: str


def load_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        cohere_rerank_model=os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
    )
