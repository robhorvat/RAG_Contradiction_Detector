from pathlib import Path

import pytest

from src.evaluation.quality_gate import GateResult, format_gate_summary, gate_passed, load_gate_result


def test_gate_passed_matrix():
    assert gate_passed(GateResult(status="pass", candidate_model=None, reason=None, checks=[], thresholds={}))
    assert not gate_passed(GateResult(status="pending", candidate_model=None, reason=None, checks=[], thresholds={}))
    assert gate_passed(
        GateResult(status="pending", candidate_model=None, reason=None, checks=[], thresholds={}),
        allow_pending=True,
    )
    assert not gate_passed(GateResult(status="fail", candidate_model=None, reason=None, checks=[], thresholds={}))
    assert gate_passed(
        GateResult(status="fail", candidate_model=None, reason=None, checks=[], thresholds={}),
        allow_fail=True,
    )


def test_load_gate_result_reads_expected_fields(tmp_path: Path):
    report = tmp_path / "eval_report.json"
    report.write_text(
        """
{
  "quality_gate": {
    "status": "FAIL",
    "candidate_model": "torch_verifier",
    "reason": "metrics below threshold",
    "thresholds": {"macro_f1_min": 0.7},
    "checks": [{"name": "macro_f1_min", "actual": 0.24, "threshold": 0.7, "passed": false}]
  }
}
        """.strip(),
        encoding="utf-8",
    )

    result = load_gate_result(report)
    assert result.status == "fail"
    assert result.candidate_model == "torch_verifier"
    assert result.reason == "metrics below threshold"
    assert result.thresholds["macro_f1_min"] == 0.7
    assert result.checks[0]["name"] == "macro_f1_min"


def test_load_gate_result_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_gate_result(tmp_path / "missing.json")


def test_format_gate_summary_contains_status_and_checks():
    result = GateResult(
        status="fail",
        candidate_model="torch_verifier",
        reason="below threshold",
        thresholds={"macro_f1_min": 0.7},
        checks=[{"name": "macro_f1_min", "actual": 0.2, "threshold": 0.7, "passed": False}],
    )
    text = format_gate_summary(result)
    assert "Quality gate status: fail" in text
    assert "Candidate model: torch_verifier" in text
    assert "macro_f1_min" in text
