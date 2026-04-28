import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from src.utils.logger import logging
from src.utils.exception import CustomException
from pathlib import Path
import os

class ModelEvaluator:
    def __init__(self, report_dir: Path):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)

    def evaluate_model(self, y_true, y_pred, y_proba, prefix="test"):
        try:
            logging.info(f"Calculating {prefix} performance metrics")
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred)),
                "recall": float(recall_score(y_true, y_pred)),
                "f1_score": float(f1_score(y_true, y_pred)),
                "roc_auc": float(roc_auc_score(y_true, y_proba))
            }
            
            self.save_plots(y_true, y_pred, y_proba, prefix)
            return metrics
        except Exception as e:
            raise CustomException(e, sys)

    def save_plots(self, y_true, y_pred, y_proba, prefix):
        try:
            # 1. Confusion Matrix (Manual plot with matplotlib)
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix - {prefix.capitalize()}')
            plt.colorbar()
            tick_marks = np.arange(2)
            plt.xticks(tick_marks, ['Benign', 'Malware'], rotation=45)
            plt.yticks(tick_marks, ['Benign', 'Malware'])
            
            # Fill matrix with values
            thresh = cm.max() / 2.
            for i, j in np.ndindex(cm.shape):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(self.report_dir / f"{prefix}_metrics.png") # Renamed as requested generic name
            plt.close()

            # 2. ROC Curve
            plt.figure(figsize=(8, 6))
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_proba):.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {prefix.capitalize()}')
            plt.legend()
            plt.savefig(self.report_dir / f"{prefix}_roc_curve.png")
            plt.close()
            
            logging.info(f"Plots saved for {prefix} in {self.report_dir}")
        except Exception as e:
            raise CustomException(e, sys)
