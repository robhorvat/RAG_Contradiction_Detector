from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def build_bootstrap_report() -> dict:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "project": "RAG_Contradiction_Detector",
        "run_type": "bootstrap_template",
        "generated_at_utc": generated_at,
        "data": {
            "eval_split": "tbd",
            "n_samples": None,
            "notes": "Replace placeholders once evaluation harness is implemented.",
        },
        "metrics": {
            "retrieval": {
                "recall_at_k": None,
                "mrr_at_k": None,
            },
            "verdict": {
                "macro_f1": None,
                "contradiction_f1": None,
            },
        },
        "baselines": {
            "heuristic_verdict_macro_f1": None,
            "llm_only_verdict_macro_f1": None,
        },
        "quality_gate": {
            "target_macro_f1_min": 0.70,
            "target_delta_over_heuristic_min": 0.10,
            "status": "pending",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a bootstrap evaluation report template.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/eval_report.bootstrap.json"),
        help="Path for the generated report JSON.",
    )
    args = parser.parse_args()

    payload = build_bootstrap_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved bootstrap report: {args.output}")


if __name__ == "__main__":
    main()
