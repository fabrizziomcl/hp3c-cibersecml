from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    preprocessor_obj_file_path: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_file_path: Path
    base_model_path: Path
    params_n_estimators: int
    params_max_depth: int
