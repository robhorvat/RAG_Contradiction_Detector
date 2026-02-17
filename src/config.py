from __future__ import annotations

from dataclasses import dataclass
import os


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return bool(default)
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    openai_api_key: str | None
    cohere_api_key: str | None
    gemini_api_key: str | None
    openai_chat_model: str
    gemini_model: str
    cohere_rerank_model: str
    verifier_backend: str
    verifier_strategy: str
    verifier_override_confidence: float
    torch_verifier_checkpoint: str | None
    model_registry_latest_path: str
    metrics_enabled: bool
    metrics_host: str
    metrics_port: int


def load_settings() -> Settings:
    try:
        override_confidence = float(os.getenv("VERIFIER_OVERRIDE_CONFIDENCE", "0.65"))
    except ValueError:
        override_confidence = 0.65
    try:
        metrics_port = int(os.getenv("METRICS_PORT", "9108"))
    except ValueError:
        metrics_port = 9108
    metrics_port = max(1, min(65535, metrics_port))

    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "openai").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        cohere_rerank_model=os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0"),
        verifier_backend=os.getenv("VERIFIER_BACKEND", "heuristic").strip().lower(),
        verifier_strategy=os.getenv("VERIFIER_STRATEGY", "confidence_override").strip().lower(),
        verifier_override_confidence=max(0.0, min(1.0, override_confidence)),
        torch_verifier_checkpoint=os.getenv("TORCH_VERIFIER_CHECKPOINT"),
        model_registry_latest_path=os.getenv("MODEL_REGISTRY_LATEST_PATH", "artifacts/model_registry_latest.json"),
        metrics_enabled=_parse_bool(os.getenv("METRICS_ENABLED", "1"), default=True),
        metrics_host=os.getenv("METRICS_HOST", "0.0.0.0").strip(),
        metrics_port=metrics_port,
    )
