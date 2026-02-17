from src.verifier.heuristic_verifier import HeuristicContradictionVerifier


def test_predict_contradictory_when_overlap_and_opposite_polarity():
    verifier = HeuristicContradictionVerifier(min_overlap_ratio=0.10)
    out = verifier.predict(
        "Vitamin D supplementation reduces hip fracture incidence in elderly adults.",
        "Vitamin D supplementation does not reduce hip fracture incidence in elderly adults.",
    )
    assert out["verdict"] == "Contradictory"
    assert 0.0 <= out["confidence"] <= 1.0


def test_predict_supporting_when_overlap_and_same_polarity():
    verifier = HeuristicContradictionVerifier(min_overlap_ratio=0.10)
    out = verifier.predict(
        "Higher vitamin D dose reduces nonvertebral fractures in older adults.",
        "A higher vitamin D dose reduced nonvertebral fracture risk in adults over 65.",
    )
    assert out["verdict"] == "Supporting"
    assert out["overlap_ratio"] >= 0.10


def test_predict_unrelated_when_low_overlap():
    verifier = HeuristicContradictionVerifier(min_overlap_ratio=0.20)
    out = verifier.predict(
        "Vitamin D supplementation reduces nonvertebral fracture risk in older adults.",
        "Coffee consumption was associated with cardiovascular mortality in the cohort.",
    )
    assert out["verdict"] == "Unrelated"
    assert out["overlap_ratio"] < 0.20
