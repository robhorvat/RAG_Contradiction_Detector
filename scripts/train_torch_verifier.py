from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.verifier.torch_nli_verifier import (
    NLI_LABELS,
    PairNLIClassifier,
    TorchNLIVerifier,
    TorchVerifierConfig,
    build_pair_batch,
)

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        "Training requires a working torch installation. "
        "Install CPU build with: "
        "`python -m pip install --index-url https://download.pytorch.org/whl/cpu torch`"
    ) from exc


LABEL_TO_ID = {label: i for i, label in enumerate(NLI_LABELS)}


def load_pairs(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("label") not in LABEL_TO_ID:
                continue
            rows.append(row)
    return rows


def _json_safe_args(namespace: argparse.Namespace) -> dict:
    out = {}
    for k, v in vars(namespace).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def sample_rows(rows: list[dict], max_samples: int | None, rng: random.Random) -> list[dict]:
    if not max_samples or max_samples <= 0 or len(rows) <= max_samples:
        return rows
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    keep = set(idx[: max_samples])
    return [row for i, row in enumerate(rows) if i in keep]


def iter_batches(rows: list[dict], batch_size: int, rng: random.Random):
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield [rows[i] for i in idx[start : start + batch_size]]


def _safe_div(n: float, d: float) -> float:
    return 0.0 if d == 0 else float(n / d)


def macro_f1(y_true: list[int], y_pred: list[int], n_labels: int = 3) -> tuple[float, dict]:
    per_label = {}
    f1_values = []
    for label_id in range(n_labels):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label_id and yp == label_id)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label_id and yp == label_id)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label_id and yp != label_id)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_label[NLI_LABELS[label_id]] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for yt in y_true if yt == label_id),
        }
        f1_values.append(f1)
    return float(sum(f1_values) / len(f1_values)), per_label


def evaluate(model: PairNLIClassifier, rows: list[dict], cfg: TorchVerifierConfig, device: torch.device) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    total = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for start in range(0, len(rows), 128):
            batch = rows[start : start + 128]
            pairs = [(r["claim_text"], r["evidence_text"]) for r in batch]
            labels = torch.as_tensor([LABEL_TO_ID[r["label"]] for r in batch], dtype=torch.long, device=device)
            ids_a, mask_a, ids_b, mask_b = build_pair_batch(
                pairs,
                vocab_size=cfg.vocab_size,
                max_tokens=cfg.max_tokens,
                device=device,
            )
            logits = model(ids_a, mask_a, ids_b, mask_b)
            loss = loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            total_loss += float(loss.item()) * int(labels.size(0))
            total += int(labels.size(0))

    acc = _safe_div(sum(int(a == b) for a, b in zip(y_true, y_pred)), len(y_true))
    m_f1, per_label = macro_f1(y_true, y_pred, n_labels=len(NLI_LABELS))
    return {
        "loss": round(_safe_div(total_loss, total), 6),
        "accuracy": round(acc, 4),
        "macro_f1": round(m_f1, 4),
        "per_label": per_label,
        "n_samples": int(total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the torch NLI verifier on SciFact claim-evidence pairs.")
    parser.add_argument("--train-file", type=Path, default=Path("data/scifact/processed/train_pairs.jsonl"))
    parser.add_argument("--dev-file", type=Path, default=Path("data/scifact/processed/dev_pairs.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/torch_verifier"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=0)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--vocab-size", type=int, default=30000)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    train_rows = load_pairs(args.train_file)
    dev_rows = load_pairs(args.dev_file)

    if not train_rows or not dev_rows:
        raise SystemExit(
            "Training/dev pairs missing or empty. Run `python scripts/prepare_scifact_pairs.py` first."
        )

    train_rows = sample_rows(train_rows, int(args.max_train_samples), rng)
    dev_rows = sample_rows(dev_rows, int(args.max_dev_samples), rng)

    cfg = TorchVerifierConfig(
        vocab_size=int(args.vocab_size),
        max_tokens=int(args.max_tokens),
        embed_dim=int(args.embed_dim),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        num_labels=len(NLI_LABELS),
    )

    verifier = TorchNLIVerifier.build_fresh(cfg)
    device = verifier.device
    model = verifier.model
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_dev_f1 = -math.inf
    best_state = None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for batch in iter_batches(train_rows, int(args.batch_size), rng):
            pairs = [(r["claim_text"], r["evidence_text"]) for r in batch]
            labels = torch.as_tensor([LABEL_TO_ID[r["label"]] for r in batch], dtype=torch.long, device=device)
            ids_a, mask_a, ids_b, mask_b = build_pair_batch(
                pairs,
                vocab_size=cfg.vocab_size,
                max_tokens=cfg.max_tokens,
                device=device,
            )
            logits = model(ids_a, mask_a, ids_b, mask_b)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * int(labels.size(0))
            seen += int(labels.size(0))

        train_loss = _safe_div(running_loss, seen)
        dev_metrics = evaluate(model, dev_rows, cfg, device)
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "dev_loss": dev_metrics["loss"],
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"dev_loss={row['dev_loss']:.4f} dev_acc={row['dev_accuracy']:.4f} "
            f"dev_macro_f1={row['dev_macro_f1']:.4f}"
        )

        if dev_metrics["macro_f1"] > best_dev_f1:
            best_dev_f1 = float(dev_metrics["macro_f1"])
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final_dev = evaluate(model, dev_rows, cfg, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = args.output_dir / "torch_nli_verifier.pt"
    report_path = args.output_dir / "train_report.json"

    torch.save(
        {
            "config": cfg.to_dict(),
            "labels": list(NLI_LABELS),
            "model_state": model.state_dict(),
            "train_args": _json_safe_args(args),
            "history": history,
            "final_dev": final_dev,
            "device_used": str(device),
        },
        ckpt_path,
    )

    report = {
        "status": "ok",
        "checkpoint": str(ckpt_path),
        "device_used": str(device),
        "train_samples": len(train_rows),
        "dev_samples": len(dev_rows),
        "history": history,
        "final_dev": final_dev,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
