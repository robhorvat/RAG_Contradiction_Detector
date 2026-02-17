from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_registry import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Show recent model registry entries.")
    parser.add_argument("--registry", type=Path, default=Path("artifacts/model_registry.jsonl"))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    rows = read_jsonl(args.registry)
    if not rows:
        print(f"No registry entries found at: {args.registry}")
        return

    print(f"Registry: {args.registry}")
    print(f"Total entries: {len(rows)}")
    print(f"Showing last {min(len(rows), int(args.limit))}:")
    for row in rows[-int(args.limit) :]:
        print(
            f"- run_id={row.get('run_id')} "
            f"macro_f1={row.get('metrics', {}).get('dev_macro_f1')} "
            f"acc={row.get('metrics', {}).get('dev_accuracy')} "
            f"device={row.get('runtime', {}).get('device_used')}"
        )


if __name__ == "__main__":
    main()
