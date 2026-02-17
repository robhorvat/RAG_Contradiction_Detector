from __future__ import annotations

VALID_VERDICTS = {"Contradictory", "Supporting", "Unrelated"}


def _normalize_verdict(value: str) -> str:
    return value if value in VALID_VERDICTS else "Unrelated"


def arbitrate_verdict(
    *,
    llm_verdict: str,
    verifier_prediction: dict | None,
    strategy: str = "llm_only",
    override_confidence: float = 0.65,
) -> tuple[str, dict]:
    """
    Decide the final verdict from LLM output and optional verifier output.

    Strategies:
    - llm_only: always use LLM verdict.
    - verifier_only: use verifier verdict when available.
    - confidence_override: use verifier verdict only when it disagrees and
      confidence >= override_confidence.
    """
    llm_verdict_norm = _normalize_verdict(llm_verdict)
    selected_verdict = llm_verdict_norm
    selected_source = "llm"
    override_applied = False
    reason = "llm_only strategy selected"
    verifier_verdict = None
    verifier_confidence = None

    if verifier_prediction:
        verifier_verdict = _normalize_verdict(str(verifier_prediction.get("verdict", "Unrelated")))
        try:
            verifier_confidence = float(verifier_prediction.get("confidence", 0.0))
        except (TypeError, ValueError):
            verifier_confidence = 0.0
    else:
        verifier_confidence = 0.0

    strategy_norm = (strategy or "llm_only").strip().lower()

    if strategy_norm == "verifier_only":
        if verifier_prediction is None:
            reason = "verifier_only requested but verifier output missing"
        else:
            selected_verdict = verifier_verdict
            selected_source = "torch_verifier"
            override_applied = selected_verdict != llm_verdict_norm
            reason = "verifier_only strategy selected"
    elif strategy_norm == "confidence_override":
        if verifier_prediction is None:
            reason = "confidence_override requested but verifier output missing"
        elif verifier_verdict == llm_verdict_norm:
            reason = "llm and verifier agree"
        elif verifier_confidence >= float(override_confidence):
            selected_verdict = verifier_verdict
            selected_source = "torch_verifier"
            override_applied = True
            reason = "verifier confidence above threshold and disagrees with llm"
        else:
            reason = "verifier confidence below threshold; keeping llm verdict"
    else:
        strategy_norm = "llm_only"

    details = {
        "strategy": strategy_norm,
        "override_confidence": round(float(override_confidence), 4),
        "llm_verdict": llm_verdict_norm,
        "verifier_verdict": verifier_verdict,
        "verifier_confidence": None if verifier_prediction is None else round(float(verifier_confidence), 4),
        "selected_verdict": selected_verdict,
        "selected_source": selected_source,
        "override_applied": bool(override_applied),
        "reason": reason,
    }
    return selected_verdict, details
