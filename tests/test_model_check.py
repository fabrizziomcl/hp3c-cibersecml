"""Gate de promoción por recall (model_check)."""

import json
from pathlib import Path

from src.utils.model_check import check_improvement


def _escribir_corrida(reports_dir: Path, nombre: str, recall: float):
    run = reports_dir / nombre
    run.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "model_performance": {"test": {"recall": recall}},
    }
    (run / "report.json").write_text(json.dumps(payload))


def test_sin_corridas_aprueba(tmp_path):
    assert check_improvement(tmp_path) is True


def test_primera_corrida_aprueba(tmp_path):
    _escribir_corrida(tmp_path, "run_20260101_000000", 0.90)
    assert check_improvement(tmp_path) is True


def test_regresion_de_recall_rechaza(tmp_path):
    _escribir_corrida(tmp_path, "run_20260101_000000", 0.90)
    _escribir_corrida(tmp_path, "run_20260102_000000", 0.80)
    assert check_improvement(tmp_path) is False


def test_mejora_de_recall_aprueba(tmp_path):
    _escribir_corrida(tmp_path, "run_20260101_000000", 0.80)
    _escribir_corrida(tmp_path, "run_20260102_000000", 0.95)
    assert check_improvement(tmp_path) is True


def test_reporte_con_esquema_invalido_rechaza(tmp_path):
    run = tmp_path / "run_20260102_000000"
    run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps({"schema_version": 1}))
    assert check_improvement(tmp_path) is False
