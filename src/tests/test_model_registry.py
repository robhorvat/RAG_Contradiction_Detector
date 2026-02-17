from pathlib import Path

from src.model_registry import append_jsonl, read_jsonl, sha256_file, write_json


def test_sha256_file_is_stable(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc123")
    h1 = sha256_file(p)
    h2 = sha256_file(p)
    assert h1 == h2
    assert len(h1) == 64


def test_jsonl_append_and_read_roundtrip(tmp_path):
    path = tmp_path / "registry.jsonl"
    append_jsonl(path, {"run_id": "run-1", "metric": 0.5})
    append_jsonl(path, {"run_id": "run-2", "metric": 0.7})
    rows = read_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["run_id"] == "run-1"
    assert rows[1]["metric"] == 0.7


def test_write_json_creates_parent(tmp_path):
    out = tmp_path / "nested" / "latest.json"
    write_json(out, {"ok": True})
    assert out.exists()
    assert '"ok": true' in out.read_text(encoding="utf-8").lower()
