from __future__ import annotations

import threading

HAS_PROMETHEUS = True
try:
    from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, start_http_server
except Exception:  # noqa: BLE001
    HAS_PROMETHEUS = False


_server_lock = threading.Lock()
_server_started = False
_registry = CollectorRegistry() if HAS_PROMETHEUS else None

if HAS_PROMETHEUS:
    PIPELINE_REQUESTS_TOTAL = Counter(
        "rag_pipeline_requests_total",
        "Total successful RAG contradiction checks.",
        labelnames=("llm_provider", "final_verdict", "verdict_source"),
        registry=_registry,
    )
    PIPELINE_ERRORS_TOTAL = Counter(
        "rag_pipeline_errors_total",
        "Total failed RAG contradiction checks.",
        labelnames=("llm_provider", "error_type"),
        registry=_registry,
    )
    PIPELINE_DURATION_SECONDS = Histogram(
        "rag_pipeline_duration_seconds",
        "End-to-end RAG pipeline duration in seconds.",
        labelnames=("llm_provider", "outcome"),
        registry=_registry,
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0),
    )
    RETRIEVED_CHUNKS = Histogram(
        "rag_retrieved_chunks",
        "Retrieved chunk counts per paper slot.",
        labelnames=("paper_slot",),
        registry=_registry,
        buckets=(0, 1, 2, 3, 5, 8, 12, 20, 40),
    )
    VERIFIER_OVERRIDES_TOTAL = Counter(
        "rag_verifier_overrides_total",
        "Count of verifier overrides applied during arbitration.",
        labelnames=("strategy",),
        registry=_registry,
    )
    VERIFIER_DISAGREEMENTS_TOTAL = Counter(
        "rag_verifier_disagreements_total",
        "Count of LLM vs verifier verdict disagreements.",
        labelnames=("strategy",),
        registry=_registry,
    )


def start_metrics_server(*, host: str = "0.0.0.0", port: int = 9108) -> tuple[bool, str]:
    if not HAS_PROMETHEUS:
        return False, "prometheus_client not installed"

    global _server_started  # noqa: PLW0603
    with _server_lock:
        if _server_started:
            return True, "already_running"
        try:
            start_http_server(port=int(port), addr=str(host), registry=_registry)
        except Exception as exc:  # noqa: BLE001
            return False, f"start_failed: {exc}"
        _server_started = True
        return True, "started"


def _sanitize_verdict(value: str) -> str:
    normalized = str(value or "Unrelated").strip()
    if normalized in {"Contradictory", "Supporting", "Unrelated"}:
        return normalized
    return "Unrelated"


def record_pipeline_success(
    *,
    llm_provider: str,
    final_verdict: str,
    verdict_source: str,
    duration_seconds: float,
    chunks_paper_1: int,
    chunks_paper_2: int,
    arbitration: dict | None,
    enabled: bool = True,
) -> None:
    if not HAS_PROMETHEUS or not enabled:
        return

    provider = str(llm_provider or "unknown")
    final_verdict = _sanitize_verdict(final_verdict)
    verdict_source = str(verdict_source or "llm")
    duration_seconds = max(0.0, float(duration_seconds))

    PIPELINE_REQUESTS_TOTAL.labels(
        llm_provider=provider,
        final_verdict=final_verdict,
        verdict_source=verdict_source,
    ).inc()
    PIPELINE_DURATION_SECONDS.labels(llm_provider=provider, outcome="success").observe(duration_seconds)
    RETRIEVED_CHUNKS.labels(paper_slot="paper_1").observe(max(0, int(chunks_paper_1)))
    RETRIEVED_CHUNKS.labels(paper_slot="paper_2").observe(max(0, int(chunks_paper_2)))

    arbitration = arbitration or {}
    strategy = str(arbitration.get("strategy", "unknown"))
    if arbitration.get("override_applied"):
        VERIFIER_OVERRIDES_TOTAL.labels(strategy=strategy).inc()

    verifier_verdict = arbitration.get("verifier_verdict")
    llm_verdict = arbitration.get("llm_verdict")
    if verifier_verdict and llm_verdict and str(verifier_verdict) != str(llm_verdict):
        VERIFIER_DISAGREEMENTS_TOTAL.labels(strategy=strategy).inc()


def record_pipeline_error(
    *,
    llm_provider: str,
    error_type: str,
    duration_seconds: float,
    enabled: bool = True,
) -> None:
    if not HAS_PROMETHEUS or not enabled:
        return

    provider = str(llm_provider or "unknown")
    err = str(error_type or "unknown_error")
    duration_seconds = max(0.0, float(duration_seconds))

    PIPELINE_ERRORS_TOTAL.labels(llm_provider=provider, error_type=err).inc()
    PIPELINE_DURATION_SECONDS.labels(llm_provider=provider, outcome="error").observe(duration_seconds)


def render_metrics_text() -> str:
    if not HAS_PROMETHEUS:
        return ""
    return generate_latest(_registry).decode("utf-8")
