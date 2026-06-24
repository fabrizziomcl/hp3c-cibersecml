import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from src.data.transformation import DataTransformation
from src.entity.config_entity import ModelTrainerConfig
from src.models.evaluate import ModelEvaluator
from src.utils.exception import CustomException
from src.utils.hpc import resolver_workers
from src.utils.logger import logging

# Versión del esquema del reporte JSON. Debe coincidir con la versión que
# valida src/utils/model_check.py.
REPORT_SCHEMA_VERSION = 2

# Número de pliegues para la validación cruzada.
CV_FOLDS = 5

# Métrica usada como referencia en la validación cruzada y en el gate CI/CD.
CV_SCORING = "recall"


class ModelTrainer:
    """Entrena un Random Forest y persiste el modelo y un reporte versionado.

    Mide el tiempo de entrenamiento en modo secuencial (un solo núcleo) frente
    al modo paralelo (varios núcleos) para exponer métricas HPC de *speedup* y
    eficiencia, y consolida todas las métricas en un reporte JSON trazable.
    """

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = Path("reports") / f"run_{self.timestamp}"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def initiate_model_trainer(
        self, X_train, y_train, X_test, y_test, hpc_metrics, dataset_info
    ):
        """Entrena, evalúa y reporta el modelo. Devuelve el clasificador paralelo."""
        try:
            workers = resolver_workers(self.config.n_jobs)
            logging.info(
                "Entrenando RandomForest -> n_estimators=%d, max_depth=%d, "
                "n_jobs solicitado=%d, workers efectivos=%d (disponibles=%d, "
                "recorte=%s), random_state=%d",
                self.config.params_n_estimators,
                self.config.params_max_depth,
                workers.solicitados,
                workers.efectivos,
                workers.disponibles,
                workers.hubo_recorte,
                self.config.random_state,
            )

            # --- Línea base secuencial (un solo núcleo) ---
            inicio_seq = time.perf_counter()
            rf_seq = RandomForestClassifier(
                n_estimators=self.config.params_n_estimators,
                max_depth=self.config.params_max_depth,
                n_jobs=1,
                random_state=self.config.random_state,
            )
            rf_seq.fit(X_train, y_train)
            tiempo_seq = time.perf_counter() - inicio_seq

            # --- Entrenamiento paralelo (workers efectivos) ---
            inicio_par = time.perf_counter()
            rf = RandomForestClassifier(
                n_estimators=self.config.params_n_estimators,
                max_depth=self.config.params_max_depth,
                n_jobs=workers.efectivos,
                random_state=self.config.random_state,
            )
            rf.fit(X_train, y_train)
            tiempo_par = time.perf_counter() - inicio_par

            speedup = tiempo_seq / tiempo_par if tiempo_par > 0 else float("inf")
            eficiencia = speedup / workers.efectivos
            hpc_metrics_train = {
                "time_seq": tiempo_seq,
                "time_par": tiempo_par,
                "speedup": speedup,
                "efficiency": eficiencia,
                "num_workers": workers.efectivos,
                "backend": "joblib/loky",
                **workers.como_dict(),
            }
            logging.info(
                "Métricas HPC de entrenamiento: t_seq=%.4fs, t_par=%.4fs, "
                "speedup=%.2fx, eficiencia=%.2f",
                tiempo_seq,
                tiempo_par,
                speedup,
                eficiencia,
            )

            evaluator = ModelEvaluator(self.report_dir)
            test_metrics = evaluator.evaluate_model(
                y_test,
                rf.predict(X_test),
                rf.predict_proba(X_test)[:, 1],
                prefix="test",
            )
            train_metrics = evaluator.evaluate_model(
                y_train,
                rf.predict(X_train),
                rf.predict_proba(X_train)[:, 1],
                prefix="train",
            )
            logging.info("Métricas en prueba: %s", test_metrics)
            logging.info("Métricas en entrenamiento: %s", train_metrics)

            # Validación cruzada de 5 pliegues sobre el conjunto de
            # entrenamiento, paralelizada con los workers efectivos.
            cv_scores = cross_val_score(
                rf,
                X_train,
                y_train,
                cv=CV_FOLDS,
                scoring=CV_SCORING,
                n_jobs=workers.efectivos,
            )
            cv_metrics = {
                "cv_recall_mean": float(cv_scores.mean()),
                "cv_recall_std": float(cv_scores.std()),
                "cv_folds": int(cv_scores.size),
            }
            logging.info(
                "Validación cruzada (%d pliegues): recall medio=%.4f ± %.4f",
                cv_metrics["cv_folds"],
                cv_metrics["cv_recall_mean"],
                cv_metrics["cv_recall_std"],
            )

            os.makedirs(
                os.path.dirname(self.config.trained_model_file_path), exist_ok=True
            )
            joblib.dump(rf, self.config.trained_model_file_path)
            logging.info(
                "Modelo guardado en %s", self.config.trained_model_file_path
            )

            self._escribir_reporte(
                train_metrics,
                test_metrics,
                cv_metrics,
                hpc_metrics,
                hpc_metrics_train,
                dataset_info,
            )
            return rf
        except Exception as e:
            raise CustomException(e, sys)

    def _escribir_reporte(
        self,
        train_metrics,
        test_metrics,
        cv_metrics,
        hpc_metrics_trans,
        hpc_metrics_train,
        dataset_info,
    ):
        """Consolida todas las métricas en reports/run_<timestamp>/report.json."""
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "metadata": {
                "report_id": f"REP-{uuid4().hex[:8]}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": "Random Forest Classifier",
                "parameters": {
                    "n_estimators": self.config.params_n_estimators,
                    "max_depth": self.config.params_max_depth,
                    "n_jobs": self.config.n_jobs,
                    "random_state": self.config.random_state,
                    "cv_folds": CV_FOLDS,
                    "cv_scoring": CV_SCORING,
                },
            },
            "dataset": dataset_info,
            "hpc_performance": {
                "transformation": hpc_metrics_trans,
                "training": hpc_metrics_train,
            },
            "model_performance": {
                "train": train_metrics,
                "test": test_metrics,
                "cv": cv_metrics,
            },
        }

        report_path = self.report_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        logging.info("Reporte escrito en %s", report_path)


def _construir_info_dataset(y, y_train, y_test) -> dict:
    """Resume tamaños y distribución de clases del dataset y sus particiones."""
    valores, conteos = np.unique(y, return_counts=True)
    return {
        "total_rows": int(y.size),
        "class_distribution": {int(k): int(v) for k, v in zip(valores, conteos)},
        "train_size": int(y_train.size),
        "test_size": int(y_test.size),
    }


if __name__ == "__main__":
    from src.config.config import ConfigurationManager

    cfg = ConfigurationManager()
    dt_cfg = cfg.get_data_transformation_config()
    mt_cfg = cfg.get_model_trainer_config()

    X_transformado, etiquetas, hpc_metrics = DataTransformation(
        dt_cfg
    ).initiate_data_transformation()

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformado,
        etiquetas,
        test_size=mt_cfg.test_size,
        random_state=mt_cfg.random_state,
        stratify=etiquetas,
    )

    dataset_info = _construir_info_dataset(etiquetas, y_train, y_test)
    logging.info(
        "Particiones -> entrenamiento=%d, prueba=%d (total=%d)",
        y_train.size,
        y_test.size,
        etiquetas.size,
    )
    ModelTrainer(mt_cfg).initiate_model_trainer(
        X_train, y_train, X_test, y_test, hpc_metrics, dataset_info
    )
