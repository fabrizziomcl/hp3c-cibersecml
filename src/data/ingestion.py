import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.entity.config_entity import DataIngestionConfig


class DataIngestion:
    """
    Splits the master dataset into a training/eval partition (raw/) and a
    simulation partition (external/) used to feed CI-driven retraining.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion")
        try:
            source = self.config.local_data_file
            if not source.exists():
                raise FileNotFoundError(f"Master dataset not found at {source}")

            df = pd.read_csv(source)
            logging.info(f"Loaded master dataset shape={df.shape}")

            os.makedirs(self.config.train_eval_path.parent, exist_ok=True)
            os.makedirs(self.config.simulation_path.parent, exist_ok=True)

            stratify = df["Class"] if "Class" in df.columns else None
            train_eval_df, simulation_df = train_test_split(
                df,
                test_size=self.config.simulation_split_size,
                random_state=self.config.random_state,
                stratify=stratify,
            )

            train_eval_df.to_csv(self.config.train_eval_path, index=False)
            simulation_df.to_csv(self.config.simulation_path, index=False)

            logging.info(
                "Ingestion done. train_eval=%d, simulation=%d",
                len(train_eval_df),
                len(simulation_df),
            )
            return self.config.train_eval_path, self.config.simulation_path
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.config.config import ConfigurationManager

    config = ConfigurationManager().get_data_ingestion_config()
    DataIngestion(config).initiate_data_ingestion()
