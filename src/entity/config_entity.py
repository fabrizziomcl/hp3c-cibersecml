from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    train_eval_path: Path
    simulation_path: Path
    simulation_split_size: float
    random_state: int

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    preprocessor_obj_file_path: Path
    pca_components: int
    num_workers: int
    force_imbalance: bool
    random_state: int

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    trained_model_file_path: Path
    params_n_estimators: int
    params_max_depth: int
    test_size: float
    n_jobs: int
    random_state: int
