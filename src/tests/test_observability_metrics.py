import uuid

import pytest

from src.observability.metrics import (
    HAS_PROMETHEUS,
    record_pipeline_error,
    record_pipeline_success,
    render_metrics_text,
)


pytestmark = pytest.mark.skipif(not HAS_PROMETHEUS, reason="prometheus_client not installed")


def test_record_pipeline_success_exposes_labels():
    unique_provider = f"pytest-{uuid.uuid4().hex[:8]}"
    record_pipeline_success(
        llm_provider=unique_provider,
        final_verdict="Contradictory",
        verdict_source="torch_verifier",
        duration_seconds=0.42,
        chunks_paper_1=3,
        chunks_paper_2=5,
        arbitration={
            "strategy": "confidence_override",
            "llm_verdict": "Supporting",
            "verifier_verdict": "Contradictory",
            "override_applied": True,
        },
        enabled=True,
    )
    metrics = render_metrics_text()
    assert "rag_pipeline_requests_total" in metrics
    assert unique_provider in metrics
    assert "rag_verifier_overrides_total" in metrics


def test_record_pipeline_error_exposes_counter():
    unique_provider = f"pytest-{uuid.uuid4().hex[:8]}"
    record_pipeline_error(
        llm_provider=unique_provider,
        error_type="ValidationError",
        duration_seconds=0.13,
        enabled=True,
    )
    metrics = render_metrics_text()
    assert "rag_pipeline_errors_total" in metrics
    assert unique_provider in metrics
