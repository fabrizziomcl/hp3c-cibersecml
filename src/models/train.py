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
from src.utils.logger import logging

REPORT_SCHEMA_VERSION = 2


def _make_random_forest(config: ModelTrainerConfig, n_jobs: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.params_n_estimators,
        max_depth=config.params_max_depth,
        n_jobs=n_jobs,
        random_state=config.random_state,
    )


def _train_random_forest(
    X_train, y_train, config: ModelTrainerConfig, n_jobs: int
) -> RandomForestClassifier:
    rf = _make_random_forest(config, n_jobs=n_jobs)
    rf.fit(X_train, y_train)
    return rf


def _measure_random_forest_training_time(
    X_train, y_train, config: ModelTrainerConfig, n_jobs: int
) -> tuple[RandomForestClassifier, float]:
    start = time.perf_counter()
    rf = _train_random_forest(X_train, y_train, config, n_jobs=n_jobs)
    elapsed = time.perf_counter() - start
    return rf, elapsed


def _benchmark_random_forest_parallelism(
    X_train, y_train, config: ModelTrainerConfig
) -> tuple[dict, RandomForestClassifier]:
    logging.info(
        "Benchmarking RandomForest training with sequential_baseline=n_jobs=1 and parallel_n_jobs=%d",
        config.n_jobs,
    )

    _, sequential_time = _measure_random_forest_training_time(
        X_train, y_train, config, n_jobs=1
    )
    parallel_model, parallel_time = _measure_random_forest_training_time(
        X_train, y_train, config, n_jobs=config.n_jobs
    )

    effective_cores = os.cpu_count() or 1 if config.n_jobs == -1 else max(1, int(config.n_jobs))
    speedup = sequential_time / parallel_time if parallel_time > 0 else float("inf")
    efficiency = speedup / effective_cores

    logging.info(
        "RandomForest run metrics: n_jobs=%d sequential_time=%.6f parallel_time=%.6f speedup=%.4f efficiency=%.4f cores_used=%d",
        config.n_jobs,
        sequential_time,
        parallel_time,
        speedup,
        efficiency,
        effective_cores,
    )

    return (
        {
            "sequential_time": sequential_time,
            "parallel_time": parallel_time,
            "speedup": speedup,
            "efficiency": efficiency,
            "cores_used": effective_cores,
        },
        parallel_model,
    )


class ModelTrainer:
    """
    Trains a Random Forest with configurable parallelism (n_jobs) and persists
    the model + a versioned JSON report.
    """

    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = Path("reports") / f"run_{self.timestamp}"
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def initiate_model_trainer(
        self, X_train, y_train, X_test, y_test, hpc_metrics, dataset_info
    ):
        try:
            logging.info(
                "Training RandomForest n_estimators=%d max_depth=%d n_jobs=%d",
                self.config.params_n_estimators,
                self.config.params_max_depth,
                self.config.n_jobs,
            )
            training_hpc_metrics, rf = _benchmark_random_forest_parallelism(
                X_train, y_train, self.config
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

            # 5-fold CV on the training partition, fully parallelized
            cv_scores = cross_val_score(
                rf, X_train, y_train, cv=5, scoring="recall", n_jobs=self.config.n_jobs
            )
            cv_metrics = {
                "cv_recall_mean": float(cv_scores.mean()),
                "cv_recall_std": float(cv_scores.std()),
                "cv_folds": int(cv_scores.size),
            }

            logging.info("Training performance metrics: %s", train_metrics)
            logging.info("Test performance metrics: %s", test_metrics)
            logging.info("Cross-validation metrics: %s", cv_metrics)
            logging.info("Transformation HPC metrics: %s", hpc_metrics)
            logging.info("Training HPC metrics: %s", training_hpc_metrics)

            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            joblib.dump(rf, self.config.trained_model_file_path)
            logging.info("Model saved at %s", self.config.trained_model_file_path)

            self._write_report(
                train_metrics,
                test_metrics,
                cv_metrics,
                hpc_metrics,
                training_hpc_metrics,
                dataset_info,
            )
            return rf
        except Exception as e:
            raise CustomException(e, sys)

    def _write_report(
        self,
        train_metrics,
        test_metrics,
        cv_metrics,
        hpc_metrics,
        training_hpc_metrics,
        dataset_info,
    ):
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
                },
            },
            "dataset": dataset_info,
            "hpc_performance": hpc_metrics,
            "training_hpc_performance": training_hpc_metrics,
            "model_performance": {
                "train": train_metrics,
                "test": test_metrics,
                "cv": cv_metrics,
            },
        }

        report_path = self.report_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        logging.info("Report written at %s", report_path)


def _build_dataset_info(y, y_train, y_test) -> dict:
    unique, counts = np.unique(y, return_counts=True)
    return {
        "total_rows": int(y.size),
        "class_distribution": {int(k): int(v) for k, v in zip(unique, counts)},
        "train_size": int(y_train.size),
        "test_size": int(y_test.size),
    }


if __name__ == "__main__":
    from src.config.config import ConfigurationManager

    cfg = ConfigurationManager()
    dt_cfg = cfg.get_data_transformation_config()
    mt_cfg = cfg.get_model_trainer_config()

    X_transformed, labels, hpc_metrics = DataTransformation(dt_cfg).initiate_data_transformation()

    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed,
        labels,
        test_size=mt_cfg.test_size,
        random_state=mt_cfg.random_state,
        stratify=labels,
    )

    dataset_info = _build_dataset_info(labels, y_train, y_test)
    ModelTrainer(mt_cfg).initiate_model_trainer(
        X_train, y_train, X_test, y_test, hpc_metrics, dataset_info
    )
