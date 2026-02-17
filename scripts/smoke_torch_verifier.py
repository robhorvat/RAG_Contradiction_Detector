from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    try:
        from src.verifier.torch_nli_verifier import TorchNLIVerifier, TorchVerifierConfig
    except RuntimeError as exc:
        print(f"Torch smoke test skipped: {exc}")
        return

    cfg = TorchVerifierConfig(vocab_size=5000, max_tokens=24, embed_dim=32, hidden_dim=64, dropout=0.1)
    verifier = TorchNLIVerifier.build_fresh(cfg)
    out = verifier.predict(
        "Vitamin D supplementation reduced nonvertebral fracture risk in elderly adults.",
        "Vitamin D supplementation did not reduce nonvertebral fracture risk in elderly adults.",
    )
    print(
        f"Torch smoke check verdict: {out['verdict']} "
        f"(confidence={out['confidence']}, device={out['device']}, mode={out['device_mode']})"
    )


if __name__ == "__main__":
    main()
