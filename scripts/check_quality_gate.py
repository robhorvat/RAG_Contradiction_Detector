from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.quality_gate import format_gate_summary, gate_passed, load_gate_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce evaluation quality gate from report JSON.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/eval_report.json"),
        help="Path to evaluation report JSON.",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Treat pending status as pass (useful for early local development).",
    )
    parser.add_argument(
        "--allow-fail",
        action="store_true",
        help="Treat fail status as pass (soft mode for local diagnostics).",
    )
    args = parser.parse_args()

    result = load_gate_result(args.report)
    print(format_gate_summary(result))

    if gate_passed(result, allow_pending=args.allow_pending, allow_fail=args.allow_fail):
        print("Quality gate decision: PASS")
        return

    raise SystemExit("Quality gate decision: BLOCK")


if __name__ == "__main__":
    main()
