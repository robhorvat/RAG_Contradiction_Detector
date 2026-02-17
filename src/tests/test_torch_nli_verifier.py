import pytest


HAS_TORCH = True
try:
    import torch
    from src.verifier.torch_nli_verifier import (
        NLI_LABELS,
        PairNLIClassifier,
        TorchNLIVerifier,
        TorchVerifierConfig,
        build_pair_batch,
        text_to_hashed_ids,
    )
except Exception:  # noqa: BLE001
    HAS_TORCH = False


pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


def test_text_to_hashed_ids_is_deterministic():
    ids_1 = text_to_hashed_ids("Vitamin D reduces fractures in elderly adults.")
    ids_2 = text_to_hashed_ids("Vitamin D reduces fractures in elderly adults.")
    assert ids_1 == ids_2
    assert len(ids_1) > 0


def test_build_pair_batch_shapes():
    pairs = [
        ("Vitamin D reduces fractures.", "Vitamin D does not reduce fractures."),
        ("Coffee affects CVD risk.", "Coffee affects mortality risk."),
    ]
    ids_a, mask_a, ids_b, mask_b = build_pair_batch(pairs, vocab_size=1000, max_tokens=16)
    assert ids_a.shape[0] == 2
    assert ids_b.shape[0] == 2
    assert mask_a.dtype == torch.float32
    assert mask_b.dtype == torch.float32


def test_pair_classifier_forward_shape():
    model = PairNLIClassifier(vocab_size=2000, embed_dim=32, hidden_dim=64, dropout=0.1, num_labels=3)
    pairs = [("a b c", "a b d"), ("x y", "q r")]
    ids_a, mask_a, ids_b, mask_b = build_pair_batch(pairs, vocab_size=2000, max_tokens=8)
    logits = model(ids_a, mask_a, ids_b, mask_b)
    assert logits.shape == (2, 3)


def test_torch_verifier_predict_contract():
    cfg = TorchVerifierConfig(vocab_size=4000, max_tokens=24, embed_dim=32, hidden_dim=64, dropout=0.1)
    verifier = TorchNLIVerifier.build_fresh(cfg)
    out = verifier.predict(
        "Vitamin D supplementation reduces fracture incidence in older adults.",
        "Vitamin D supplementation does not reduce fracture incidence in older adults.",
    )
    assert out["verdict"] in NLI_LABELS
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["device"] in {"cpu", "cuda"}
    assert out["device_mode"] in {"cpu-auto", "cuda-auto", "cpu-forced", "cuda-forced", "cuda-requested-but-unavailable"}
    assert set(out["label_probs"].keys()) == set(NLI_LABELS)
