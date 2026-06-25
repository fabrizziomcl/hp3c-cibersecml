import os
import sys
import time

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.entity.config_entity import DataTransformationConfig
from src.utils.exception import CustomException
from src.utils.hpc import resolver_workers
from src.utils.logger import logging


# Columnas constantes detectadas en el EDA; se descartan junto con cualquier
# otra de varianza cero en el dataset concatenado.
COLUMNAS_VARIANZA_CERO_CONOCIDAS = (
    "pslist.nprocs64bit",
    "handles.nport",
    "svcscan.interactive_process_services",
)
MAPEO_CLASES = {"Benign": 0, "Malware": 1}
COLUMNAS_NO_FEATURE = ("Class", "Category")


def _escalar_fragmento(
    fragmento: np.ndarray, media: np.ndarray, escala: np.ndarray
) -> np.ndarray:
    """Escalado vectorizado de un fragmento; seguro en paralelo."""
    return (fragmento - media) / escala


class DataTransformation:
    """Concatena y limpia los CSV de ``data/raw/``, ajusta el preprocesador
    (``StandardScaler`` + ``PCA``) y mide el escalado secuencial vs paralelo."""

    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.feature_cols: list[str] = []

    def _cargar_y_limpiar(self) -> pd.DataFrame:
        raw_path = self.config.data_path
        archivos = sorted(f for f in os.listdir(raw_path) if f.endswith(".csv"))
        if not archivos:
            raise CustomException(f"No se encontraron CSV en {raw_path}", sys)
        logging.info(
            "Se encontraron %d CSV en %s: %s", len(archivos), raw_path, archivos
        )

        df = pd.concat(
            (pd.read_csv(os.path.join(raw_path, f)) for f in archivos),
            ignore_index=True,
        )
        filas_iniciales = len(df)
        logging.info("Dataset concatenado: shape=%s", df.shape)

        # Normalización de etiquetas: se conservan los dos primeros segmentos
        # de Category (p. ej. "Trojan-A1B2-..." -> "Trojan-A1B2").
        if "Category" in df.columns:
            df["Category"] = (
                df["Category"].astype(str).str.split("-").str[:2].str.join("-")
            )

        df.drop_duplicates(keep="first", inplace=True)
        logging.info(
            "Duplicados eliminados: %d filas (%d -> %d)",
            filas_iniciales - len(df),
            filas_iniciales,
            len(df),
        )

        # Elimina las columnas de varianza cero conocidas más cualquier otra
        # columna numérica cuya varianza sea exactamente cero.
        columnas_estaticas = [
            c for c in COLUMNAS_VARIANZA_CERO_CONOCIDAS if c in df.columns
        ]
        df.drop(columns=columnas_estaticas, inplace=True, errors="ignore")
        numericas = df.select_dtypes(include=[np.number])
        columnas_dinamicas = [
            c for c in numericas.columns if numericas[c].var() == 0 and c != "Class"
        ]
        if columnas_dinamicas:
            logging.info(
                "Columnas de varianza cero adicionales eliminadas: %s",
                columnas_dinamicas,
            )
            df.drop(columns=columnas_dinamicas, inplace=True)
        logging.info(
            "Columnas de varianza cero eliminadas: %d conocidas + %d dinámicas",
            len(columnas_estaticas),
            len(columnas_dinamicas),
        )

        if "Class" in df.columns:
            df["Class"] = df["Class"].map(MAPEO_CLASES).astype(int)

        if self.config.force_imbalance:
            logging.warning(
                "FORCE_IMBALANCE=True: modo experimental, submuestreando malware"
            )
            benignos = df[df["Class"] == 0]
            malware = df[df["Class"] == 1].sample(
                frac=0.2, random_state=self.config.random_state
            )
            df = pd.concat([benignos, malware]).sample(
                frac=1, random_state=self.config.random_state
            )
            logging.info(
                "Nueva distribución de clases: %s",
                df["Class"].value_counts().to_dict(),
            )

        return df

    def initiate_data_transformation(self):
        """Ejecuta la transformación completa y devuelve (X_pca, y, métricas HPC)."""
        try:
            df = self._cargar_y_limpiar()

            columna_objetivo = "Class"
            self.feature_cols = [
                c
                for c in df.select_dtypes(include=[np.number]).columns
                if c not in COLUMNAS_NO_FEATURE
            ]
            X = df[self.feature_cols].to_numpy(dtype=np.float64)
            y = df[columna_objetivo].to_numpy()
            logging.info(
                "Matriz de características: %d filas x %d columnas",
                X.shape[0],
                X.shape[1],
            )

            preprocesador: Pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=self.config.pca_components)),
                ]
            )
            preprocesador.fit(X)

            scaler: StandardScaler = preprocesador.named_steps["scaler"]
            media, escala = scaler.mean_, scaler.scale_

            workers = resolver_workers(self.config.num_workers)
            logging.info(
                "Workers para escalado HPC -> solicitados=%d, disponibles=%d, "
                "efectivos=%d (recorte=%s)",
                workers.solicitados,
                workers.disponibles,
                workers.efectivos,
                workers.hubo_recorte,
            )

            # --- Línea base secuencial: solo el escalado (comparación justa) ---
            inicio_seq = time.perf_counter()
            X_escalado_seq = _escalar_fragmento(X, media, escala)
            tiempo_seq = time.perf_counter() - inicio_seq

            # --- Paralelo: escalado por fragmentos vía joblib (backend loky) ---
            fragmentos = np.array_split(X, workers.efectivos)
            inicio_par = time.perf_counter()
            fragmentos_escalados = Parallel(
                n_jobs=workers.efectivos, backend="loky"
            )(delayed(_escalar_fragmento)(c, media, escala) for c in fragmentos)
            X_escalado_par = np.vstack(fragmentos_escalados)
            tiempo_par = time.perf_counter() - inicio_par

            # Verificación: el resultado paralelo debe ser numéricamente igual
            # al secuencial.
            if not np.allclose(X_escalado_seq, X_escalado_par, equal_nan=True):
                raise CustomException(
                    "El escalado paralelo divergió del secuencial", sys
                )

            speedup = tiempo_seq / tiempo_par if tiempo_par > 0 else float("inf")
            eficiencia = speedup / workers.efectivos
            logging.info(
                "Métricas HPC de transformación: t_seq=%.6fs, t_par=%.6fs, "
                "speedup=%.4fx, eficiencia=%.4f",
                tiempo_seq,
                tiempo_par,
                speedup,
                eficiencia,
            )

            # PCA sobre los datos (correctamente) escalados.
            X_pca = preprocesador.named_steps["pca"].transform(X_escalado_par)
            varianza_explicada = preprocesador.named_steps[
                "pca"
            ].explained_variance_ratio_
            logging.info(
                "PCA -> %d componentes, varianza explicada acumulada=%.4f",
                self.config.pca_components,
                float(varianza_explicada.sum()),
            )

            # Persistencia de artefactos.
            os.makedirs(self.config.root_dir, exist_ok=True)
            os.makedirs(self.config.preprocessor_obj_file_path.parent, exist_ok=True)
            np.save(self.config.root_dir / "transformed_data.npy", X_pca)
            np.save(self.config.root_dir / "labels.npy", y)
            joblib.dump(
                {"pipeline": preprocesador, "feature_cols": self.feature_cols},
                self.config.preprocessor_obj_file_path,
            )
            logging.info(
                "Preprocesador guardado en %s",
                self.config.preprocessor_obj_file_path,
            )

            metricas_hpc = {
                "time_seq": tiempo_seq,
                "time_par": tiempo_par,
                "speedup": speedup,
                "efficiency": eficiencia,
                "num_workers": workers.efectivos,
                "backend": "joblib/loky",
                "n_features": int(X.shape[1]),
                "pca_components": int(self.config.pca_components),
                "pca_varianza_explicada": float(varianza_explicada.sum()),
                **workers.como_dict(),
            }
            return X_pca, y, metricas_hpc
        except Exception as e:
            raise CustomException(e, sys)
