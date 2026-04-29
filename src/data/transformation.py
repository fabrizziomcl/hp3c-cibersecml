import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.entity.config_entity import DataTransformationConfig
from src.utils.exception import CustomException
from src.utils.logger import logging


# Columns that the EDA flagged as zero-variance on the master dataset. They
# are removed defensively, but we ALSO drop any other column whose variance
# is exactly zero on the current concatenated dataset.
_KNOWN_ZERO_VAR = (
    "pslist.nprocs64bit",
    "handles.nport",
    "svcscan.interactive_process_services",
)


def _scale_chunk(chunk: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply a pre-fitted scaling to a chunk. Pure numpy, parallel-safe."""
    return (chunk - mean) / scale


class DataTransformation:
    """
    Concatenates every CSV in raw/, cleans + dedupes, fits a sklearn Pipeline
    (StandardScaler + PCA) and persists it. Measures sequential vs parallel
    transform times to expose HPC metrics.
    """

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    def _load_and_clean(self) -> pd.DataFrame:
        raw_path = self.config.data_path
        all_files = sorted(f for f in os.listdir(raw_path) if f.endswith(".csv"))
        if not all_files:
            raise CustomException(f"No CSV files found in {raw_path}", sys)
        logging.info("Found %d CSVs in %s: %s", len(all_files), raw_path, all_files)

        df = pd.concat(
            (pd.read_csv(os.path.join(raw_path, f)) for f in all_files),
            ignore_index=True,
        )
        logging.info("Concatenated dataset shape=%s", df.shape)

        if "Category" in df.columns:
            df["Category"] = df["Category"].astype(str).str.split("-").str[:2].str.join("-")
        df.drop_duplicates(keep="first", inplace=True)

        # Drop hardcoded zero-variance columns + any other zero-variance numeric columns
        zero_static = [c for c in _KNOWN_ZERO_VAR if c in df.columns]
        df.drop(columns=zero_static, inplace=True, errors="ignore")
        numeric = df.select_dtypes(include=[np.number])
        zero_dynamic = [c for c in numeric.columns if numeric[c].var() == 0 and c != "Class"]
        if zero_dynamic:
            logging.info("Dropping additional zero-variance columns: %s", zero_dynamic)
            df.drop(columns=zero_dynamic, inplace=True)

        if "Class" in df.columns:
            df["Class"] = df["Class"].map({"Benign": 0, "Malware": 1}).astype(int)

        if self.config.force_imbalance:
            logging.warning("FORCE_IMBALANCE=True: experimental mode, downsampling malware")
            benign = df[df["Class"] == 0]
            malware = df[df["Class"] == 1].sample(
                frac=0.2, random_state=self.config.random_state
            )
            df = pd.concat([benign, malware]).sample(
                frac=1, random_state=self.config.random_state
            )
            logging.info("New class distribution: %s", df["Class"].value_counts().to_dict())

        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def initiate_data_transformation(self):
        try:
            df = self._load_and_clean()

            target_col = "Class"
            self.feature_cols = [
                c
                for c in df.select_dtypes(include=[np.number]).columns
                if c not in (target_col, "Category")
            ]
            X = df[self.feature_cols].to_numpy(dtype=np.float64)
            y = df[target_col].to_numpy()

            preprocessor: Pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=self.config.pca_components)),
                ]
            )
            preprocessor.fit(X)

            scaler: StandardScaler = preprocessor.named_steps["scaler"]
            mean, scale = scaler.mean_, scaler.scale_
            num_workers = max(1, int(self.config.num_workers))

            # --- Sequential baseline: scaling only (apples-to-apples) ---
            start_seq = time.perf_counter()
            X_scaled_seq = _scale_chunk(X, mean, scale)
            time_seq = time.perf_counter() - start_seq

            # --- Parallel: scaling chunked via joblib (loky backend) ---
            chunks = np.array_split(X, num_workers)
            start_par = time.perf_counter()
            scaled_chunks = Parallel(n_jobs=num_workers, backend="loky")(
                delayed(_scale_chunk)(c, mean, scale) for c in chunks
            )
            X_scaled_par = np.vstack(scaled_chunks)
            time_par = time.perf_counter() - start_par

            # Sanity check: parallel result must be numerically equal to sequential
            if not np.allclose(X_scaled_seq, X_scaled_par, equal_nan=True):
                raise CustomException("Parallel scaling diverged from sequential", sys)

            speedup = time_seq / time_par if time_par > 0 else float("inf")
            efficiency = speedup / num_workers

            # PCA on the (correctly) scaled data — use the fitted preprocessor
            X_pca = preprocessor.named_steps["pca"].transform(X_scaled_par)

            # Persist artifacts
            os.makedirs(self.config.root_dir, exist_ok=True)
            os.makedirs(self.config.preprocessor_obj_file_path.parent, exist_ok=True)
            np.save(self.config.root_dir / "transformed_data.npy", X_pca)
            np.save(self.config.root_dir / "labels.npy", y)
            joblib.dump(
                {"pipeline": preprocessor, "feature_cols": self.feature_cols},
                self.config.preprocessor_obj_file_path,
            )
            logging.info(
                "Preprocessor saved at %s", self.config.preprocessor_obj_file_path
            )

            return X_pca, y, {
                "time_seq": time_seq,
                "time_par": time_par,
                "speedup": speedup,
                "efficiency": efficiency,
                "num_workers": num_workers,
            }
        except Exception as e:
            raise CustomException(e, sys)
