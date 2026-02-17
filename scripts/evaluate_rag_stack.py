from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import compute_classification_metrics, compute_retrieval_metrics_from_rankings
from src.verifier.heuristic_verifier import HeuristicContradictionVerifier


LABELS = ("Contradictory", "Supporting", "Unrelated")
DEFAULT_K_VALUES = (1, 3, 5, 10)
TOKEN_PATTERN = re.compile(r"[a-zA-Z]{3,}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _parse_k_values(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        k = int(chunk)
        if k > 0:
            values.append(k)
    if not values:
        return DEFAULT_K_VALUES
    return tuple(sorted(set(values)))


def _tokenize(text: str) -> set[str]:
    return {tok for tok in TOKEN_PATTERN.findall((text or "").lower())}


def _doc_to_text(doc: dict) -> str:
    title = str(doc.get("title", "")).strip()
    abstract_sents = doc.get("abstract", []) or []
    abstract = " ".join(str(s).strip() for s in abstract_sents[:2] if str(s).strip())
    if title and abstract:
        return f"{title}. {abstract}"
    return abstract or title


def _lexical_score(a_tokens: set[str], b_tokens: set[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    return float(overlap) / float(max(1, min(len(a_tokens), len(b_tokens))))


def evaluate_retrieval_lexical(
    *,
    claims_path: Path,
    corpus_path: Path,
    k_values: tuple[int, ...],
    max_claims: int,
) -> dict:
    if not claims_path.exists() or not corpus_path.exists():
        return {
            "status": "skipped",
            "reason": f"missing file(s): {claims_path} or {corpus_path}",
        }

    claims = _load_jsonl(claims_path)
    corpus_rows = _load_jsonl(corpus_path)
    corpus_tokens: dict[int, set[str]] = {
        int(row["doc_id"]): _tokenize(_doc_to_text(row)) for row in corpus_rows if "doc_id" in row
    }

    rankings: list[list[int]] = []
    relevant_sets: list[set[int]] = []
    query_count = 0

    for claim in claims:
        evidence = claim.get("evidence", {}) or {}
        relevant_doc_ids = {int(doc_id) for doc_id in evidence.keys()}
        if not relevant_doc_ids:
            continue

        claim_tokens = _tokenize(str(claim.get("claim", "")))
        if not claim_tokens:
            continue

        scored_docs = []
        for doc_id, doc_tokens in corpus_tokens.items():
            score = _lexical_score(claim_tokens, doc_tokens)
            scored_docs.append((score, doc_id))

        ranking = [doc_id for _, doc_id in sorted(scored_docs, key=lambda x: (-x[0], x[1]))]
        rankings.append(ranking)
        relevant_sets.append(relevant_doc_ids)
        query_count += 1

        if max_claims > 0 and query_count >= max_claims:
            break

    metric_payload = compute_retrieval_metrics_from_rankings(
        rankings,
        relevant_sets,
        k_values=k_values,
    )
    metric_payload["status"] = "ok"
    return metric_payload


def _resolve_torch_checkpoint(explicit_path: str, registry_latest_path: Path) -> tuple[Path | None, str]:
    if explicit_path:
        ckpt = Path(explicit_path)
        if ckpt.exists():
            return ckpt, "explicit"
        return None, f"explicit checkpoint not found: {ckpt}"

    if not registry_latest_path.exists():
        return None, f"registry snapshot not found: {registry_latest_path}"

    try:
        latest = json.loads(registry_latest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, f"invalid JSON in registry snapshot: {registry_latest_path}"

    checkpoint = latest.get("artifact", {}).get("checkpoint_path")
    if not checkpoint:
        return None, f"checkpoint path missing in registry snapshot: {registry_latest_path}"

    ckpt = Path(checkpoint)
    if ckpt.exists():
        return ckpt, "registry-latest"
    return None, f"checkpoint from registry snapshot does not exist: {ckpt}"


def evaluate_verdict_models(
    *,
    pairs_path: Path,
    max_pairs: int,
    seed: int,
    torch_checkpoint: str,
    registry_latest_path: Path,
) -> tuple[dict, dict]:
    if not pairs_path.exists():
        raise SystemExit(f"Pairs file not found: {pairs_path}. Run `make prep-scifact` first.")

    rows = _load_jsonl(pairs_path)
    rows = [row for row in rows if row.get("label") in LABELS]
    if not rows:
        raise SystemExit(f"No labeled rows found in {pairs_path}.")

    if max_pairs > 0 and len(rows) > max_pairs:
        rng = random.Random(int(seed))
        rows = rng.sample(rows, k=max_pairs)

    y_true = [str(row["label"]) for row in rows]
    model_outputs: dict[str, list[str]] = {}

    model_outputs["majority_unrelated"] = ["Unrelated" for _ in rows]

    heuristic = HeuristicContradictionVerifier()
    model_outputs["heuristic"] = [
        heuristic.predict(str(row["claim_text"]), str(row["evidence_text"]))["verdict"] for row in rows
    ]

    model_meta = {
        "majority_unrelated": {"status": "ok"},
        "heuristic": {"status": "ok"},
    }

    torch_ckpt, ckpt_source = _resolve_torch_checkpoint(torch_checkpoint, registry_latest_path)
    if torch_ckpt is not None:
        try:
            from src.verifier.torch_nli_verifier import TorchNLIVerifier

            verifier = TorchNLIVerifier.from_checkpoint(torch_ckpt)
            model_outputs["torch_verifier"] = [
                verifier.predict(str(row["claim_text"]), str(row["evidence_text"]))["verdict"] for row in rows
            ]
            model_meta["torch_verifier"] = {
                "status": "ok",
                "checkpoint_path": str(torch_ckpt),
                "checkpoint_source": ckpt_source,
                "device": str(verifier.device),
                "device_mode": verifier.device_mode,
            }
        except Exception as exc:  # noqa: BLE001
            model_meta["torch_verifier"] = {
                "status": "skipped",
                "reason": f"failed to load/predict: {exc}",
            }
    else:
        model_meta["torch_verifier"] = {
            "status": "skipped",
            "reason": ckpt_source,
        }

    metrics_by_model = {}
    for model_name, y_pred in model_outputs.items():
        metrics_by_model[model_name] = compute_classification_metrics(
            y_true,
            y_pred,
            labels=LABELS,
        )
        metrics_by_model[model_name]["contradiction_f1"] = metrics_by_model[model_name]["per_label"]["Contradictory"]["f1"]

    return (
        {
            "n_pairs": len(rows),
            "labels": list(LABELS),
            "models": metrics_by_model,
        },
        model_meta,
    )


def evaluate_quality_gate(
    *,
    verdict_metrics: dict,
    model_meta: dict,
    macro_f1_min: float,
    delta_over_heuristic_min: float,
) -> dict:
    models = verdict_metrics.get("models", {})
    heuristic = models.get("heuristic")
    torch_metrics = models.get("torch_verifier")
    torch_meta = model_meta.get("torch_verifier", {})

    checks = []
    if torch_metrics is None or torch_meta.get("status") != "ok":
        return {
            "status": "pending",
            "candidate_model": "torch_verifier",
            "reason": "torch verifier metrics unavailable",
            "thresholds": {
                "macro_f1_min": round(float(macro_f1_min), 4),
                "delta_over_heuristic_min": round(float(delta_over_heuristic_min), 4),
            },
            "checks": checks,
        }

    macro_f1 = float(torch_metrics.get("macro_f1", 0.0))
    heuristic_macro_f1 = float(heuristic.get("macro_f1", 0.0)) if heuristic else 0.0
    delta = macro_f1 - heuristic_macro_f1

    check_macro = {
        "name": "macro_f1_min",
        "actual": round(macro_f1, 4),
        "threshold": round(float(macro_f1_min), 4),
        "passed": macro_f1 >= float(macro_f1_min),
    }
    check_delta = {
        "name": "delta_over_heuristic_min",
        "actual": round(delta, 4),
        "threshold": round(float(delta_over_heuristic_min), 4),
        "passed": delta >= float(delta_over_heuristic_min),
    }
    checks.extend([check_macro, check_delta])

    return {
        "status": "pass" if all(c["passed"] for c in checks) else "fail",
        "candidate_model": "torch_verifier",
        "reason": "evaluated torch verifier against thresholds",
        "thresholds": {
            "macro_f1_min": round(float(macro_f1_min), 4),
            "delta_over_heuristic_min": round(float(delta_over_heuristic_min), 4),
        },
        "checks": checks,
    }


def _format_report_markdown(payload: dict) -> str:
    retrieval = payload.get("metrics", {}).get("retrieval", {})
    verdict = payload.get("metrics", {}).get("verdict", {})
    models = verdict.get("models", {})
    quality_gate = payload.get("quality_gate", {})

    lines = [
        "# Evaluation Report",
        "",
        f"- Generated at (UTC): `{payload.get('generated_at_utc')}`",
        f"- Verdict pairs evaluated: `{verdict.get('n_pairs', 0)}`",
        "",
        "## Retrieval Metrics (Lexical Baseline)",
    ]

    if retrieval.get("status") == "ok":
        lines.append(f"- Queries evaluated: `{retrieval.get('n_queries', 0)}`")
        lines.append("- Recall@k:")
        for k, val in retrieval.get("recall_at_k", {}).items():
            lines.append(f"  - k={k}: `{val}`")
        lines.append("- MRR@k:")
        for k, val in retrieval.get("mrr_at_k", {}).items():
            lines.append(f"  - k={k}: `{val}`")
    else:
        lines.append(f"- Status: `{retrieval.get('status', 'unknown')}`")
        lines.append(f"- Reason: `{retrieval.get('reason', 'n/a')}`")

    lines.extend(
        [
            "",
            "## Verdict Metrics",
            "",
            "| Model | Accuracy | Macro-F1 | Contradiction-F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for model_name in ("majority_unrelated", "heuristic", "torch_verifier"):
        m = models.get(model_name)
        if not m:
            continue
        lines.append(
            "| "
            f"{model_name} | {m.get('accuracy', 'n/a')} | {m.get('macro_f1', 'n/a')} | "
            f"{m.get('contradiction_f1', 'n/a')} |"
        )

    lines.extend(["", "## Quality Gate"])
    lines.append(f"- Status: `{quality_gate.get('status', 'unknown')}`")
    lines.append(f"- Candidate model: `{quality_gate.get('candidate_model', 'n/a')}`")
    for check in quality_gate.get("checks", []):
        lines.append(
            "- "
            f"{check.get('name')}: actual `{check.get('actual')}` vs threshold `{check.get('threshold')}` "
            f"=> `{'PASS' if check.get('passed') else 'FAIL'}`"
        )
    if quality_gate.get("reason"):
        lines.append(f"- Reason: `{quality_gate['reason']}`")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible retrieval+verdict evaluation for the RAG contradiction stack.")
    parser.add_argument("--pairs-file", type=Path, default=Path("data/scifact/processed/dev_pairs.jsonl"))
    parser.add_argument("--scifact-claims-file", type=Path, default=Path("data/scifact/raw/data/claims_dev.jsonl"))
    parser.add_argument("--scifact-corpus-file", type=Path, default=Path("data/scifact/raw/data/corpus.jsonl"))
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional cap for verdict rows (0 = use all).")
    parser.add_argument("--max-claims", type=int, default=0, help="Optional cap for retrieval claims (0 = use all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", type=str, default="1,3,5,10")
    parser.add_argument("--torch-checkpoint", type=str, default="")
    parser.add_argument("--registry-latest", type=Path, default=Path("artifacts/model_registry_latest.json"))
    parser.add_argument("--macro-f1-min", type=float, default=0.70)
    parser.add_argument("--delta-over-heuristic-min", type=float, default=0.10)
    parser.add_argument("--output-json", type=Path, default=Path("reports/eval_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/eval_report.md"))
    args = parser.parse_args()

    k_values = _parse_k_values(args.k_values)
    retrieval_metrics = evaluate_retrieval_lexical(
        claims_path=args.scifact_claims_file,
        corpus_path=args.scifact_corpus_file,
        k_values=k_values,
        max_claims=int(args.max_claims),
    )
    verdict_metrics, model_meta = evaluate_verdict_models(
        pairs_path=args.pairs_file,
        max_pairs=int(args.max_pairs),
        seed=int(args.seed),
        torch_checkpoint=args.torch_checkpoint,
        registry_latest_path=args.registry_latest,
    )
    quality_gate = evaluate_quality_gate(
        verdict_metrics=verdict_metrics,
        model_meta=model_meta,
        macro_f1_min=float(args.macro_f1_min),
        delta_over_heuristic_min=float(args.delta_over_heuristic_min),
    )

    payload = {
        "project": "RAG_Contradiction_Detector",
        "run_type": "eval_harness_v1",
        "generated_at_utc": _utc_now_iso(),
        "data": {
            "pairs_file": str(args.pairs_file),
            "scifact_claims_file": str(args.scifact_claims_file),
            "scifact_corpus_file": str(args.scifact_corpus_file),
            "max_pairs": int(args.max_pairs),
            "max_claims": int(args.max_claims),
            "seed": int(args.seed),
            "k_values": list(k_values),
        },
        "metrics": {
            "retrieval": retrieval_metrics,
            "verdict": verdict_metrics,
        },
        "model_meta": model_meta,
        "quality_gate": quality_gate,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.output_md.write_text(_format_report_markdown(payload), encoding="utf-8")

    print(f"Saved eval report JSON: {args.output_json}")
    print(f"Saved eval report Markdown: {args.output_md}")
    print(
        "Quality gate status: "
        f"{quality_gate.get('status')} "
        f"(candidate={quality_gate.get('candidate_model')})"
    )


if __name__ == "__main__":
    main()
