from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.transformation import DataTransformation, _escalar_fragmento
from src.entity.config_entity import DataTransformationConfig


def _crear_csv_sintetico(path: Path, n_filas: int = 200, semilla: int = 0):
    """Genera un CSV sintético con una columna de varianza cero a descartar."""
    rng = np.random.default_rng(semilla)
    df = pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, n_filas),
            "feat_b": rng.normal(5, 2, n_filas),
            "feat_c": rng.normal(-3, 0.5, n_filas),
            "feat_zero": np.zeros(n_filas),  # varianza cero, debe eliminarse
            "Category": ["Benign-x"] * (n_filas // 2)
            + ["Trojan-x"] * (n_filas - n_filas // 2),
            "Class": ["Benign"] * (n_filas // 2)
            + ["Malware"] * (n_filas - n_filas // 2),
        }
    )
    df.to_csv(path, index=False)


def test_escalar_fragmento_coincide_con_normalizacion_manual():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (50, 3))
    media = X.mean(axis=0)
    escala = X.std(axis=0)
    salida = _escalar_fragmento(X, media, escala)
    esperado = (X - media) / escala
    assert np.allclose(salida, esperado)


def test_transformation_persiste_preprocesador(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _crear_csv_sintetico(raw / "a.csv")

    cfg = DataTransformationConfig(
        root_dir=tmp_path / "processed",
        data_path=raw,
        preprocessor_obj_file_path=tmp_path / "models" / "preprocessor.pkl",
        pca_components=2,
        num_workers=2,
        force_imbalance=False,
        random_state=42,
    )

    X_pca, y, hpc = DataTransformation(cfg).initiate_data_transformation()

    assert X_pca.shape[1] == 2
    assert X_pca.shape[0] == y.size
    assert set(np.unique(y).tolist()) <= {0, 1}
    assert hpc["num_workers"] == 2
    assert cfg.preprocessor_obj_file_path.exists()

    bundle = joblib.load(cfg.preprocessor_obj_file_path)
    assert "pipeline" in bundle and "feature_cols" in bundle
    assert "feat_zero" not in bundle["feature_cols"]


def test_paralelo_coincide_con_secuencial(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _crear_csv_sintetico(raw / "a.csv", n_filas=300)
    cfg = DataTransformationConfig(
        root_dir=tmp_path / "processed",
        data_path=raw,
        preprocessor_obj_file_path=tmp_path / "models" / "preprocessor.pkl",
        pca_components=2,
        num_workers=4,
        force_imbalance=False,
        random_state=42,
    )
    # La verificación interna con np.allclose lanza si el paralelo diverge del
    # secuencial.
    DataTransformation(cfg).initiate_data_transformation()
