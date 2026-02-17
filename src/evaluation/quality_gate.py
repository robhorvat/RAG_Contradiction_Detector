from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class GateResult:
    status: str
    candidate_model: str | None
    reason: str | None
    checks: list[dict]
    thresholds: dict


def load_gate_result(report_path: Path) -> GateResult:
    if not report_path.exists():
        raise FileNotFoundError(f"Evaluation report not found: {report_path}")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    gate = payload.get("quality_gate") or {}

    status_raw = str(gate.get("status", "unknown")).strip().lower()
    if status_raw not in {"pass", "fail", "pending"}:
        status_raw = "unknown"

    checks = gate.get("checks") or []
    if not isinstance(checks, list):
        checks = []

    thresholds = gate.get("thresholds") or {}
    if not isinstance(thresholds, dict):
        thresholds = {}

    return GateResult(
        status=status_raw,
        candidate_model=gate.get("candidate_model"),
        reason=gate.get("reason"),
        checks=checks,
        thresholds=thresholds,
    )


def gate_passed(
    result: GateResult,
    *,
    allow_pending: bool = False,
    allow_fail: bool = False,
) -> bool:
    if result.status == "pass":
        return True
    if result.status == "pending":
        return bool(allow_pending)
    if result.status == "fail":
        return bool(allow_fail)
    return False


def format_gate_summary(result: GateResult) -> str:
    lines = [
        f"Quality gate status: {result.status}",
        f"Candidate model: {result.candidate_model or 'n/a'}",
    ]
    if result.reason:
        lines.append(f"Reason: {result.reason}")

    if result.thresholds:
        lines.append("Thresholds:")
        for name, value in result.thresholds.items():
            lines.append(f"  - {name}: {value}")

    if result.checks:
        lines.append("Checks:")
        for check in result.checks:
            lines.append(
                "  - "
                f"{check.get('name', 'unknown')}: actual={check.get('actual')} "
                f"threshold={check.get('threshold')} passed={check.get('passed')}"
            )

    return "\n".join(lines)
