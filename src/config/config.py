import os
from pathlib import Path
from dotenv import load_dotenv
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

# Load environment variables once at import time. If .env is missing, defaults
# defined in get_env() take effect.
load_dotenv()


def get_env(key: str, default):
    """Read an env var, casting to the type of `default`."""
    value = os.getenv(key)
    if value is None or value == "":
        return default
    if isinstance(default, bool):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


class ConfigurationManager:
    def __init__(self, root_dir: Path | None = None):
        self.root_dir = Path(root_dir or os.getcwd())

    def _abs(self, rel: str) -> Path:
        return self.root_dir / rel

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        raw_dir = self._abs(get_env("RAW_DATA_DIR", "data/raw"))
        external_dir = self._abs(get_env("EXTERNAL_DATA_DIR", "data/external"))
        return DataIngestionConfig(
            root_dir=raw_dir,
            local_data_file=self._abs(
                get_env("DATASET_PATH", "dataset/Obfuscated-MalMem2022.csv")
            ),
            train_eval_path=raw_dir / "train_eval.csv",
            simulation_path=external_dir / "new_data_simulation.csv",
            simulation_split_size=get_env("SIMULATION_SPLIT_SIZE", 0.2),
            random_state=get_env("RANDOM_STATE", 42),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig(
            root_dir=self._abs(get_env("PROCESSED_DATA_DIR", "data/processed")),
            data_path=self._abs(get_env("RAW_DATA_DIR", "data/raw")),
            preprocessor_obj_file_path=self._abs(
                get_env("PREPROCESSOR_PATH", "models/preprocessor.pkl")
            ),
            pca_components=get_env("PCA_COMPONENTS", 3),
            num_workers=get_env("HPC_NUM_WORKERS", 4),
            force_imbalance=get_env("FORCE_IMBALANCE", False),
            random_state=get_env("RANDOM_STATE", 42),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        return ModelTrainerConfig(
            root_dir=self._abs("models"),
            trained_model_file_path=self._abs(
                get_env("MODEL_PATH", "models/model.pkl")
            ),
            params_n_estimators=get_env("RF_N_ESTIMATORS", 100),
            params_max_depth=get_env("RF_MAX_DEPTH", 12),
            test_size=get_env("TEST_SIZE", 0.2),
            n_jobs=get_env("N_JOBS", -1),
            random_state=get_env("RANDOM_STATE", 42),
        )
