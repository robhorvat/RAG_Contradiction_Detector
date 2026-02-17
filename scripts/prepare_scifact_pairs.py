from __future__ import annotations

import argparse
import json
import random
import tarfile
import urllib.request
from pathlib import Path


SCIFACT_TARBALL_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

LABEL_MAP = {
    "SUPPORT": "Supporting",
    "CONTRADICT": "Contradictory",
}


def download_scifact_if_needed(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    tar_path = raw_dir / "data.tar.gz"
    extracted_root = raw_dir / "data"

    if not tar_path.exists():
        print(f"Downloading SciFact dataset: {SCIFACT_TARBALL_URL}")
        urllib.request.urlretrieve(SCIFACT_TARBALL_URL, tar_path)  # noqa: S310

    if not extracted_root.exists():
        print(f"Extracting: {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(path=raw_dir)  # noqa: S202

    return extracted_root


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _join_sentences(sentences: list[str], max_sentences: int | None = None) -> str:
    if max_sentences is not None:
        sentences = sentences[: max(1, int(max_sentences))]
    return " ".join(s.strip() for s in sentences if s and s.strip())


def _doc_text_for_unrelated(doc: dict) -> str:
    title = str(doc.get("title", "")).strip()
    abstract = doc.get("abstract", []) or []
    abstract_text = _join_sentences(abstract, max_sentences=2)
    if title and abstract_text:
        return f"{title}. {abstract_text}"
    return abstract_text or title or "No document text available."


def build_pairs_for_split(
    *,
    claims: list[dict],
    corpus_by_id: dict[int, dict],
    split_name: str,
    rng: random.Random,
    max_unrelated_per_claim: int = 1,
) -> list[dict]:
    pairs: list[dict] = []
    all_doc_ids = list(corpus_by_id.keys())

    for claim_obj in claims:
        claim_id = int(claim_obj["id"])
        claim_text = str(claim_obj["claim"]).strip()
        evidence_by_doc: dict = claim_obj.get("evidence", {}) or {}
        cited_doc_ids = [int(x) for x in claim_obj.get("cited_doc_ids", [])]

        evidence_doc_ids = {int(doc_id) for doc_id in evidence_by_doc.keys()}

        # Positive/negative supervised pairs from evidence annotations.
        for doc_id_str, evidence_items in evidence_by_doc.items():
            doc_id = int(doc_id_str)
            doc = corpus_by_id.get(doc_id)
            if doc is None:
                continue

            sentences = doc.get("abstract", []) or []
            for ev in evidence_items:
                raw_label = str(ev.get("label", "")).upper()
                label = LABEL_MAP.get(raw_label)
                if label is None:
                    continue

                sent_idx = [int(i) for i in ev.get("sentences", []) if isinstance(i, int) or str(i).isdigit()]
                evidence_text = _join_sentences(
                    [sentences[i] for i in sent_idx if 0 <= i < len(sentences)],
                    max_sentences=None,
                )
                if not evidence_text:
                    evidence_text = _join_sentences(sentences, max_sentences=3)

                pairs.append(
                    {
                        "pair_id": f"{split_name}-claim{claim_id}-doc{doc_id}-lbl{label}-{len(pairs)}",
                        "split": split_name,
                        "claim_id": claim_id,
                        "doc_id": doc_id,
                        "claim_text": claim_text,
                        "evidence_text": evidence_text,
                        "label": label,
                        "source": "scifact_evidence",
                    }
                )

        # Unrelated pairs from cited docs without evidence, then random docs.
        unrelated_candidates = [doc_id for doc_id in cited_doc_ids if doc_id not in evidence_doc_ids]
        if len(unrelated_candidates) < max_unrelated_per_claim:
            blocked = set(cited_doc_ids) | evidence_doc_ids
            fallback_pool = [doc_id for doc_id in all_doc_ids if doc_id not in blocked]
            need = max_unrelated_per_claim - len(unrelated_candidates)
            if fallback_pool and need > 0:
                unrelated_candidates.extend(rng.sample(fallback_pool, k=min(need, len(fallback_pool))))

        for doc_id in unrelated_candidates[: max(0, int(max_unrelated_per_claim))]:
            doc = corpus_by_id.get(int(doc_id))
            if doc is None:
                continue
            pairs.append(
                {
                    "pair_id": f"{split_name}-claim{claim_id}-doc{doc_id}-lblUnrelated-{len(pairs)}",
                    "split": split_name,
                    "claim_id": claim_id,
                    "doc_id": int(doc_id),
                    "claim_text": claim_text,
                    "evidence_text": _doc_text_for_unrelated(doc),
                    "label": "Unrelated",
                    "source": "scifact_unrelated",
                }
            )

    return pairs


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize_labels(rows: list[dict]) -> dict:
    counts = {"Contradictory": 0, "Supporting": 0, "Unrelated": 0}
    for row in rows:
        label = row.get("label")
        if label in counts:
            counts[label] += 1
    counts["total"] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SciFact claim-evidence pairs for NLI verifier training.")
    parser.add_argument("--data-root", type=Path, default=Path("data/scifact"), help="Root data directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-unrelated-per-claim", type=int, default=1, help="Unrelated examples per claim.")
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    raw_root = download_scifact_if_needed(args.data_root / "raw")

    corpus_rows = load_jsonl(raw_root / "corpus.jsonl")
    train_claims = load_jsonl(raw_root / "claims_train.jsonl")
    dev_claims = load_jsonl(raw_root / "claims_dev.jsonl")

    corpus_by_id = {int(row["doc_id"]): row for row in corpus_rows}
    train_pairs = build_pairs_for_split(
        claims=train_claims,
        corpus_by_id=corpus_by_id,
        split_name="train",
        rng=rng,
        max_unrelated_per_claim=int(args.max_unrelated_per_claim),
    )
    dev_pairs = build_pairs_for_split(
        claims=dev_claims,
        corpus_by_id=corpus_by_id,
        split_name="dev",
        rng=rng,
        max_unrelated_per_claim=int(args.max_unrelated_per_claim),
    )

    processed_dir = args.data_root / "processed"
    write_jsonl(processed_dir / "train_pairs.jsonl", train_pairs)
    write_jsonl(processed_dir / "dev_pairs.jsonl", dev_pairs)

    summary = {
        "source": "SciFact",
        "seed": int(args.seed),
        "max_unrelated_per_claim": int(args.max_unrelated_per_claim),
        "train": summarize_labels(train_pairs),
        "dev": summarize_labels(dev_pairs),
    }
    (processed_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Prepared SciFact pairs:")
    print(json.dumps(summary, indent=2))
    print(f"Train file: {processed_dir / 'train_pairs.jsonl'}")
    print(f"Dev file: {processed_dir / 'dev_pairs.jsonl'}")


if __name__ == "__main__":
    main()
