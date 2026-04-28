import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import logging
from src.utils.exception import CustomException
from pathlib import Path

class DataIngestion:
    def __init__(self, raw_data_path: Path):
        self.raw_data_path = raw_data_path
        self.train_data_path = Path("data/raw/train_eval.csv")
        self.simulation_data_path = Path("data/external/new_data_simulation.csv")

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(self.raw_data_path)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.simulation_data_path), exist_ok=True)

            # Separate a partition for CI/CD simulation (e.g., 10% of the data)
            train_eval_df, simulation_df = train_test_split(df, test_size=0.1, random_state=42)
            
            logging.info("Saving split datasets for simulation")
            train_eval_df.to_csv(self.train_data_path, index=False, header=True)
            simulation_df.to_csv(self.simulation_data_path, index=False, header=True)

            logging.info(f"Ingestion is completed. Train/Eval size: {len(train_eval_df)}, Simulation size: {len(simulation_df)}")

            return (
                self.train_data_path,
                self.simulation_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion(Path("data/raw/Obfuscated-MalMem2022.csv"))
    obj.initiate_data_ingestion()
