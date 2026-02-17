from src.verifier.verdict_arbitration import arbitrate_verdict


def test_llm_only_keeps_llm_verdict():
    final_verdict, details = arbitrate_verdict(
        llm_verdict="Contradictory",
        verifier_prediction={"verdict": "Supporting", "confidence": 0.95},
        strategy="llm_only",
        override_confidence=0.65,
    )
    assert final_verdict == "Contradictory"
    assert details["selected_source"] == "llm"
    assert not details["override_applied"]


def test_confidence_override_uses_verifier_when_confident():
    final_verdict, details = arbitrate_verdict(
        llm_verdict="Supporting",
        verifier_prediction={"verdict": "Contradictory", "confidence": 0.91},
        strategy="confidence_override",
        override_confidence=0.65,
    )
    assert final_verdict == "Contradictory"
    assert details["selected_source"] == "torch_verifier"
    assert details["override_applied"]


def test_confidence_override_keeps_llm_when_verifier_below_threshold():
    final_verdict, details = arbitrate_verdict(
        llm_verdict="Supporting",
        verifier_prediction={"verdict": "Contradictory", "confidence": 0.42},
        strategy="confidence_override",
        override_confidence=0.65,
    )
    assert final_verdict == "Supporting"
    assert details["selected_source"] == "llm"
    assert not details["override_applied"]


def test_verifier_only_falls_back_when_prediction_missing():
    final_verdict, details = arbitrate_verdict(
        llm_verdict="Supporting",
        verifier_prediction=None,
        strategy="verifier_only",
        override_confidence=0.65,
    )
    assert final_verdict == "Supporting"
    assert details["selected_source"] == "llm"
    assert not details["override_applied"]
