import json
from pathlib import Path

from src.utils.model_check import check_improvement


def _write_run(reports_dir: Path, name: str, recall: float, schema=2):
    run = reports_dir / name
    run.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": schema,
        "model_performance": {"test": {"recall": recall}},
    }
    (run / "report.json").write_text(json.dumps(payload))


def test_no_runs_approves(tmp_path):
    assert check_improvement(tmp_path) is True


def test_first_run_approves(tmp_path):
    _write_run(tmp_path, "run_20260101_000000", 0.9)
    assert check_improvement(tmp_path) is True


def test_recall_regression_rejects(tmp_path):
    _write_run(tmp_path, "run_20260101_000000", 0.9)
    _write_run(tmp_path, "run_20260102_000000", 0.8)
    assert check_improvement(tmp_path) is False


def test_recall_improvement_accepts(tmp_path):
    _write_run(tmp_path, "run_20260101_000000", 0.8)
    _write_run(tmp_path, "run_20260102_000000", 0.95)
    assert check_improvement(tmp_path) is True


def test_invalid_schema_rejects(tmp_path):
    _write_run(tmp_path, "run_20260102_000000", 0.95, schema=1)
    assert check_improvement(tmp_path) is False


def test_skips_incompatible_old_run(tmp_path):
    # Newest with v2 + intermediate with v1 + older with v2 -> compares to older v2
    _write_run(tmp_path, "run_20260101_000000", 0.7, schema=2)
    _write_run(tmp_path, "run_20260102_000000", 0.95, schema=1)
    _write_run(tmp_path, "run_20260103_000000", 0.8, schema=2)
    assert check_improvement(tmp_path) is True
