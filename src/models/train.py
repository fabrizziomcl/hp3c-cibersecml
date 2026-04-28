import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.data.transformation import DataTransformation
from src.models.evaluate import ModelEvaluator
from pathlib import Path
import joblib

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = Path(f"reports/run_{self.timestamp}")
        os.makedirs(self.report_dir, exist_ok=True)

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test, hpc_metrics, dataset_info):
        try:
            logging.info("Training the Random Forest model...")
            rf = RandomForestClassifier(
                n_estimators=self.config.params_n_estimators,
                max_depth=self.config.params_max_depth,
                random_state=42
            )
            rf.fit(X_train, y_train)
            
            # Evaluate on Test
            evaluator = ModelEvaluator(self.report_dir)
            y_pred_test = rf.predict(X_test)
            y_proba_test = rf.predict_proba(X_test)[:, 1]
            test_metrics = evaluator.evaluate_model(y_test, y_pred_test, y_proba_test, prefix="test")
            
            # Evaluate on Train (to show learning curves/plots)
            y_pred_train = rf.predict(X_train)
            y_proba_train = rf.predict_proba(X_train)[:, 1]
            train_metrics = evaluator.evaluate_model(y_train, y_pred_train, y_proba_train, prefix="train")
            
            # Save the model
            os.makedirs(os.path.dirname(self.config.trained_model_file_path), exist_ok=True)
            joblib.dump(rf, self.config.trained_model_file_path)
            logging.info(f"Model saved at {self.config.trained_model_file_path}")
            
            # Generate JSON Report
            self.generate_json_report(train_metrics, test_metrics, hpc_metrics, dataset_info)
            
            return rf
        except Exception as e:
            raise CustomException(e, sys)

    def generate_json_report(self, train_metrics, test_metrics, hpc_metrics, dataset_info):
        report_data = {
            "metadata": {
                "report_id": f"REP-{random.randint(1000, 9999)}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": "Random Forest Classifier",
                "parameters": {
                    "n_estimators": self.config.params_n_estimators,
                    "max_depth": self.config.params_max_depth
                }
            },
            "dataset": dataset_info,
            "hpc_performance": hpc_metrics,
            "model_performance": {
                "train": train_metrics,
                "test": test_metrics
            }
        }
        
        report_path = self.report_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
        
        logging.info(f"JSON report generated at {report_path}")
        print(f"Report directory: {self.report_dir}")

if __name__ == "__main__":
    from src.config.config import ConfigurationManager
    config_mgr = ConfigurationManager()
    
    # 1. Transform Data
    dt_config = config_mgr.get_data_transformation_config()
    dt = DataTransformation(dt_config.data_path)
    X_transformed, labels, hpc_metrics = dt.initiate_data_transformation()
    
    # 2. Split for Training/Evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, labels, test_size=0.2, random_state=42)
    
    # 3. Dataset Info
    unique, counts = np.unique(labels, return_counts=True)
    dist = dict(zip([int(i) for i in unique], [int(i) for i in counts]))
    dataset_info = {
        'total_rows': len(labels),
        'class_distribution': dist,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    # 4. Train Model
    mt_config = config_mgr.get_model_trainer_config()
    mt = ModelTrainer(mt_config)
    mt.initiate_model_trainer(X_train, y_train, X_test, y_test, hpc_metrics, dataset_info)
