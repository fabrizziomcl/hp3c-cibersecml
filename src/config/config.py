import os
from pathlib import Path
from src.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig

class ConfigurationManager:
    def __init__(self):
        self.root_dir = Path(os.getcwd())
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            root_dir=self.root_dir / "data" / "raw",
            source_URL="", 
            local_data_file=self.root_dir / "dataset" / "Obfuscated-MalMem2022.csv",
            unzip_dir=self.root_dir / "data" / "raw"
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig(
            root_dir=self.root_dir / "data" / "processed",
            data_path=self.root_dir / "data" / "raw",
            preprocessor_obj_file_path=self.root_dir / "models" / "preprocessor.pkl"
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        return ModelTrainerConfig(
            root_dir=self.root_dir / "models",
            trained_model_file_path=self.root_dir / "models" / "model.pkl",
            base_model_path=self.root_dir / "models" / "model.pkl",
            params_n_estimators=100,
            params_max_depth=10
        )
