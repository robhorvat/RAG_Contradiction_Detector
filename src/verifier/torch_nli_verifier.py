from __future__ import annotations

from dataclasses import dataclass
import hashlib


try:
    import torch
    import torch.nn as nn
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        "Torch NLI verifier requires a working CPU-compatible `torch` install. "
        "Recommended fix: `uv pip uninstall torch && "
        "uv pip install --index-url https://download.pytorch.org/whl/cpu torch`."
    ) from exc


NLI_LABELS = ("Contradictory", "Supporting", "Unrelated")


def _stable_hash_token(token: str, vocab_size: int) -> int:
    digest = hashlib.blake2s(token.encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) % int(vocab_size)


def text_to_hashed_ids(
    text: str,
    *,
    vocab_size: int = 30000,
    max_tokens: int = 64,
    min_token_len: int = 2,
) -> list[int]:
    """
    Convert text into deterministic hashed token IDs.
    """
    tokens = []
    current = []
    for ch in (text or "").lower():
        if ch.isalpha():
            current.append(ch)
        else:
            if len(current) >= min_token_len:
                tokens.append("".join(current))
            current = []
    if len(current) >= min_token_len:
        tokens.append("".join(current))

    ids = [_stable_hash_token(tok, vocab_size=vocab_size) for tok in tokens[:max_tokens]]
    return ids if ids else [0]


def build_pair_batch(
    pairs: list[tuple[str, str]],
    *,
    vocab_size: int = 30000,
    max_tokens: int = 64,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build padded token ID tensors and masks for (claim_a, claim_b) pairs.
    """
    if not pairs:
        raise ValueError("pairs must not be empty")

    ids_a = [text_to_hashed_ids(a, vocab_size=vocab_size, max_tokens=max_tokens) for a, _ in pairs]
    ids_b = [text_to_hashed_ids(b, vocab_size=vocab_size, max_tokens=max_tokens) for _, b in pairs]
    max_len_a = max(len(x) for x in ids_a)
    max_len_b = max(len(x) for x in ids_b)

    pad_a = torch.zeros((len(pairs), max_len_a), dtype=torch.long, device=device)
    pad_b = torch.zeros((len(pairs), max_len_b), dtype=torch.long, device=device)
    mask_a = torch.zeros((len(pairs), max_len_a), dtype=torch.float32, device=device)
    mask_b = torch.zeros((len(pairs), max_len_b), dtype=torch.float32, device=device)

    for i, (row_a, row_b) in enumerate(zip(ids_a, ids_b)):
        la, lb = len(row_a), len(row_b)
        pad_a[i, :la] = torch.as_tensor(row_a, dtype=torch.long, device=device)
        pad_b[i, :lb] = torch.as_tensor(row_b, dtype=torch.long, device=device)
        mask_a[i, :la] = 1.0
        mask_b[i, :lb] = 1.0

    return pad_a, mask_a, pad_b, mask_b


class MeanPoolingSentenceEncoder(nn.Module):
    def __init__(self, vocab_size: int = 30000, embed_dim: int = 96, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(token_ids)  # [B, T, D]
        mask_3d = mask.unsqueeze(-1)  # [B, T, 1]
        summed = (emb * mask_3d).sum(dim=1)  # [B, D]
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
        pooled = summed / denom
        return self.dropout(pooled)


class PairNLIClassifier(nn.Module):
    """
    Small trainable NLI verifier over two sentence embeddings.
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        embed_dim: int = 96,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        num_labels: int = 3,
    ):
        super().__init__()
        self.encoder = MeanPoolingSentenceEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        feature_dim = embed_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(
        self,
        ids_a: torch.Tensor,
        mask_a: torch.Tensor,
        ids_b: torch.Tensor,
        mask_b: torch.Tensor,
    ) -> torch.Tensor:
        emb_a = self.encoder(ids_a, mask_a)
        emb_b = self.encoder(ids_b, mask_b)
        features = torch.cat(
            [emb_a, emb_b, torch.abs(emb_a - emb_b), emb_a * emb_b],
            dim=-1,
        )
        return self.classifier(features)


@dataclass
class TorchVerifierConfig:
    vocab_size: int = 30000
    max_tokens: int = 64
    embed_dim: int = 96
    hidden_dim: int = 128
    dropout: float = 0.2
    num_labels: int = 3


class TorchNLIVerifier:
    """
    Inference wrapper around PairNLIClassifier for easy integration.
    """

    def __init__(
        self,
        model: PairNLIClassifier,
        *,
        config: TorchVerifierConfig | None = None,
        device: torch.device | None = None,
    ):
        self.config = config or TorchVerifierConfig()
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def build_fresh(cls, config: TorchVerifierConfig | None = None) -> "TorchNLIVerifier":
        cfg = config or TorchVerifierConfig()
        model = PairNLIClassifier(
            vocab_size=cfg.vocab_size,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            num_labels=cfg.num_labels,
        )
        return cls(model=model, config=cfg, device=torch.device("cpu"))

    def predict(self, claim_1: str, claim_2: str) -> dict:
        ids_a, mask_a, ids_b, mask_b = build_pair_batch(
            [(claim_1, claim_2)],
            vocab_size=self.config.vocab_size,
            max_tokens=self.config.max_tokens,
            device=self.device,
        )
        with torch.no_grad():
            logits = self.model(ids_a, mask_a, ids_b, mask_b)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
            pred_idx = int(torch.argmax(logits, dim=-1).item())

        return {
            "verdict": NLI_LABELS[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "label_probs": {label: round(float(probs[i]), 4) for i, label in enumerate(NLI_LABELS)},
        }
