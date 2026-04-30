import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.exception import CustomException
from src.utils.logger import logging


class InferencePipeline:
    """
    Loads model + preprocessor lazily on first use, then reuses them. Apply the
    same scaler+PCA fitted at training time so prediction matches training space.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor_path: Optional[Path] = None,
    ):
        self.model_path = Path(
            model_path or os.getenv("MODEL_PATH", "models/model.pkl")
        )
        self.preprocessor_path = Path(
            preprocessor_path
            or os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
        )
        self._model = None
        self._preprocessor = None
        self._feature_cols: list[str] | None = None

    # ------------------------------------------------------------------
    def load(self) -> None:
        if self._model is None:
            logging.info("Loading model from %s", self.model_path)
            self._model = joblib.load(self.model_path)
        if self._preprocessor is None and self.preprocessor_path.exists():
            logging.info("Loading preprocessor from %s", self.preprocessor_path)
            bundle = joblib.load(self.preprocessor_path)
            self._preprocessor = bundle["pipeline"]
            self._feature_cols = bundle["feature_cols"]

    def _prepare(self, features) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            df = features.drop(columns=["Class", "Category"], errors="ignore")
            if self._feature_cols is not None:
                missing = [c for c in self._feature_cols if c not in df.columns]
                if missing:
                    raise CustomException(
                        f"Missing required feature columns: {missing}", sys
                    )
                df = df[self._feature_cols]
            X = df.to_numpy(dtype=np.float64)
        else:
            X = np.asarray(features, dtype=np.float64)
            if X.ndim == 1:
                X = X.reshape(1, -1)

        if self._preprocessor is not None:
            X = self._preprocessor.transform(X)
        return X

    # ------------------------------------------------------------------
    def predict(self, features) -> np.ndarray:
        try:
            self.load()
            X = self._prepare(features)
            return self._model.predict(X)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, features) -> np.ndarray:
        try:
            self.load()
            X = self._prepare(features)
            return self._model.predict_proba(X)
        except Exception as e:
            raise CustomException(e, sys)
