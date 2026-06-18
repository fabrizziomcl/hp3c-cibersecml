import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.entity.config_entity import ModelTrainerConfig
from src.models.train import ModelTrainer, _benchmark_random_forest_parallelism


def _make_training_data(n_rows: int = 40):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (n_rows, 6))
    y = np.array([0, 1] * (n_rows // 2))
    return X, y


def test_random_forest_parallel_benchmark_matches_predictions():
    X, y = _make_training_data()
    cfg = ModelTrainerConfig(
        root_dir=Path("models"),
        trained_model_file_path=Path("models") / "model.pkl",
        params_n_estimators=10,
        params_max_depth=4,
        test_size=0.2,
        n_jobs=-1,
        random_state=42,
    )

    metrics, model = _benchmark_random_forest_parallelism(X, y, cfg)

    assert model.predict(X).shape == y.shape
    assert metrics["sequential_time"] > 0
    assert metrics["parallel_time"] > 0
    assert metrics["speedup"] > 0
    assert metrics["efficiency"] > 0
    assert metrics["cores_used"] >= 1


def test_random_forest_parallel_benchmark_logs_run_metrics(caplog):
    X, y = _make_training_data()
    cfg = ModelTrainerConfig(
        root_dir=Path("models"),
        trained_model_file_path=Path("models") / "model.pkl",
        params_n_estimators=10,
        params_max_depth=4,
        test_size=0.2,
        n_jobs=-1,
        random_state=42,
    )

    with caplog.at_level("INFO"):
        _benchmark_random_forest_parallelism(X, y, cfg)

    log_text = "\n".join(record.message for record in caplog.records)
    assert "RandomForest run metrics:" in log_text
    assert "n_jobs=-1" in log_text
    assert "speedup=" in log_text
    assert "efficiency=" in log_text
    assert "sequential_time=" in log_text


def test_trainer_writes_training_hpc_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    X, y = _make_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cfg = ModelTrainerConfig(
        root_dir=tmp_path / "models",
        trained_model_file_path=tmp_path / "models" / "model.pkl",
        params_n_estimators=10,
        params_max_depth=4,
        test_size=0.2,
        n_jobs=-1,
        random_state=42,
    )

    trainer = ModelTrainer(cfg)
    trainer.initiate_model_trainer(
        X_train,
        y_train,
        X_test,
        y_test,
        hpc_metrics={
            "sequential_time": 1.0,
            "parallel_time": 0.5,
            "speedup": 2.0,
            "efficiency": 1.0,
            "num_workers": 4,
        },
        dataset_info={"total_rows": int(y.size)},
    )

    report_dirs = sorted((tmp_path / "reports").glob("run_*"))
    assert report_dirs
    report_path = report_dirs[-1] / "report.json"
    report = json.loads(report_path.read_text())

    assert "training_hpc_performance" in report
    assert report["training_hpc_performance"]["sequential_time"] > 0
    assert report["training_hpc_performance"]["parallel_time"] > 0
    assert report["model_performance"]["test"]["recall"] >= 0