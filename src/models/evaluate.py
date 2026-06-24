import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.exception import CustomException
from src.utils.logger import logging

# Etiquetas de las clases para los gráficos.
ETIQUETAS_CLASES = ["Benign", "Malware"]


class ModelEvaluator:
    """Calcula métricas de clasificación y genera artefactos gráficos
    (matriz de confusión y curva ROC) para los conjuntos indicados."""

    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

    def evaluate_model(self, y_true, y_pred, y_proba, prefix="test"):
        """Calcula las métricas de un conjunto y guarda sus gráficos.

        Args:
            y_true: etiquetas reales.
            y_pred: etiquetas predichas.
            y_proba: probabilidad de la clase positiva (malware).
            prefix: prefijo del conjunto ("train" o "test") usado en los
                nombres de archivo de los gráficos.

        Returns:
            Diccionario con accuracy, precision, recall, f1_score y roc_auc.
        """
        try:
            logging.info("Calculando métricas de rendimiento del conjunto '%s'", prefix)
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred)),
                "recall": float(recall_score(y_true, y_pred)),
                "f1_score": float(f1_score(y_true, y_pred)),
                "roc_auc": float(roc_auc_score(y_true, y_proba)),
            }

            self._guardar_graficos(y_true, y_pred, y_proba, prefix)
            return metrics
        except Exception as e:
            raise CustomException(e, sys)

    def _guardar_graficos(self, y_true, y_pred, y_proba, prefix):
        """Genera y guarda la matriz de confusión y la curva ROC."""
        try:
            # 1. Matriz de confusión.
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title(f"Matriz de confusión - {prefix.capitalize()}")
            plt.colorbar()
            marcas = np.arange(len(ETIQUETAS_CLASES))
            plt.xticks(marcas, ETIQUETAS_CLASES, rotation=45)
            plt.yticks(marcas, ETIQUETAS_CLASES)

            # Rellena cada celda con su valor numérico.
            umbral = cm.max() / 2.0
            for i, j in np.ndindex(cm.shape):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > umbral else "black",
                )

            plt.ylabel("Etiqueta real")
            plt.xlabel("Etiqueta predicha")
            plt.tight_layout()
            plt.savefig(self.report_dir / f"{prefix}_metrics.png")
            plt.close()

            # 2. Curva ROC.
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            plt.plot(fpr, tpr, label=f"Curva ROC (AUC = {auc:.2f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("Tasa de falsos positivos")
            plt.ylabel("Tasa de verdaderos positivos")
            plt.title(f"Curva ROC - {prefix.capitalize()}")
            plt.legend()
            plt.savefig(self.report_dir / f"{prefix}_roc_curve.png")
            plt.close()

            logging.info("Gráficos guardados para '%s' en %s", prefix, self.report_dir)
        except Exception as e:
            raise CustomException(e, sys)
