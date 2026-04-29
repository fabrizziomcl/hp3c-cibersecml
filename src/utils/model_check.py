import json
import sys
from pathlib import Path
from typing import Optional

REQUIRED_SCHEMA_VERSION = 2


def _load_report(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("schema_version") != REQUIRED_SCHEMA_VERSION:
        return None
    if "model_performance" not in data or "test" not in data["model_performance"]:
        return None
    if "recall" not in data["model_performance"]["test"]:
        return None
    return data


def _list_runs(reports_dir: Path = Path("reports")) -> list[Path]:
    if not reports_dir.exists():
        return []
    return sorted(
        (d for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith("run_")),
        reverse=True,
    )


def check_improvement(reports_dir: Path = Path("reports")) -> bool:
    runs = _list_runs(reports_dir)
    if not runs:
        print("INFO: No runs found. Approving by default.")
        return True

    new_data = _load_report(runs[0] / "report.json")
    if new_data is None:
        print(f"ERROR: Latest report at {runs[0]} is missing or schema-invalid.")
        return False

    old_data = next(
        (loaded for d in runs[1:] if (loaded := _load_report(d / "report.json"))),
        None,
    )
    if old_data is None:
        print("INFO: No previous compatible report. Approving by default.")
        return True

    new_recall = new_data["model_performance"]["test"]["recall"]
    old_recall = old_data["model_performance"]["test"]["recall"]
    print(f"DEBUG: new_recall={new_recall:.4f} | previous_recall={old_recall:.4f}")

    if new_recall >= old_recall:
        print("SUCCESS: Recall did not regress. Promoting model.")
        return True
    print("WARNING: Recall regression detected. Skipping promotion.")
    return False


if __name__ == "__main__":
    sys.exit(0 if check_improvement() else 1)
