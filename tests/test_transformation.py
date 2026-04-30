from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.transformation import DataTransformation, _scale_chunk
from src.entity.config_entity import DataTransformationConfig


def _make_synthetic_csv(path: Path, n_rows: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, n_rows),
            "feat_b": rng.normal(5, 2, n_rows),
            "feat_c": rng.normal(-3, 0.5, n_rows),
            "feat_zero": np.zeros(n_rows),  # zero variance, must be dropped
            "Category": ["Benign-x"] * (n_rows // 2) + ["Trojan-x"] * (n_rows - n_rows // 2),
            "Class": ["Benign"] * (n_rows // 2) + ["Malware"] * (n_rows - n_rows // 2),
        }
    )
    df.to_csv(path, index=False)


def test_scale_chunk_matches_manual_normalization():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (50, 3))
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    out = _scale_chunk(X, mean, scale)
    expected = (X - mean) / scale
    assert np.allclose(out, expected)


def test_transformation_persists_preprocessor(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_synthetic_csv(raw / "a.csv")

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


def test_parallel_matches_sequential(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_synthetic_csv(raw / "a.csv", n_rows=300)
    cfg = DataTransformationConfig(
        root_dir=tmp_path / "processed",
        data_path=raw,
        preprocessor_obj_file_path=tmp_path / "models" / "preprocessor.pkl",
        pca_components=2,
        num_workers=4,
        force_imbalance=False,
        random_state=42,
    )
    # The internal np.allclose check raises if parallel diverges from sequential
    DataTransformation(cfg).initiate_data_transformation()
