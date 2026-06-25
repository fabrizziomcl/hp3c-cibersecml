"""Lectura tipada de variables de entorno y construcción de configuraciones."""

from src.config.config import ConfigurationManager, get_env


def test_get_env_convierte_int(monkeypatch):
    monkeypatch.setenv("X_INT", "42")
    assert get_env("X_INT", 0) == 42


def test_get_env_convierte_float(monkeypatch):
    monkeypatch.setenv("X_FLOAT", "0.25")
    assert get_env("X_FLOAT", 0.0) == 0.25


def test_get_env_convierte_bool(monkeypatch):
    monkeypatch.setenv("X_BOOL", "true")
    assert get_env("X_BOOL", False) is True
    monkeypatch.setenv("X_BOOL", "0")
    assert get_env("X_BOOL", True) is False


def test_get_env_usa_default_si_falta(monkeypatch):
    monkeypatch.delenv("X_MISSING", raising=False)
    assert get_env("X_MISSING", "fallback") == "fallback"


def test_configuration_manager_valores_por_defecto(tmp_path):
    cfg = ConfigurationManager(root_dir=tmp_path)
    ingest = cfg.get_data_ingestion_config()
    assert ingest.simulation_split_size == 0.2
    assert ingest.train_eval_path == tmp_path / "data" / "raw" / "train_eval.csv"

    train = cfg.get_model_trainer_config()
    assert train.test_size == 0.2
    assert train.n_jobs == 16
