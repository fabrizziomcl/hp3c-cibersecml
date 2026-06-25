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
    """Capa de inferencia compartida por la API y la UI. Carga modelo y
    preprocesador de forma perezosa y aplica el mismo escalador + PCA del
    entrenamiento."""

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

    def load(self) -> None:
        """Carga el modelo y el preprocesador desde disco si aún no lo están."""
        if self._model is None:
            logging.info("Cargando modelo desde %s", self.model_path)
            self._model = joblib.load(self.model_path)
        if self._preprocessor is None and self.preprocessor_path.exists():
            logging.info("Cargando preprocesador desde %s", self.preprocessor_path)
            bundle = joblib.load(self.preprocessor_path)
            self._preprocessor = bundle["pipeline"]
            self._feature_cols = bundle["feature_cols"]

    def _preparar(self, features) -> np.ndarray:
        """Selecciona las columnas requeridas y aplica el preprocesador."""
        if isinstance(features, pd.DataFrame):
            df = features.drop(columns=["Class", "Category"], errors="ignore")
            if self._feature_cols is not None:
                faltantes = [c for c in self._feature_cols if c not in df.columns]
                if faltantes:
                    raise CustomException(
                        f"Faltan columnas de características requeridas: {faltantes}",
                        sys,
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

    def predict(self, features) -> np.ndarray:
        """Devuelve las etiquetas predichas para las características dadas."""
        try:
            self.load()
            X = self._preparar(features)
            return self._model.predict(X)
        except Exception as e:
            raise CustomException(e, sys)

    def predict_proba(self, features) -> np.ndarray:
        """Devuelve las probabilidades de clase para las características dadas."""
        try:
            self.load()
            X = self._preparar(features)
            return self._model.predict_proba(X)
        except Exception as e:
            raise CustomException(e, sys)
