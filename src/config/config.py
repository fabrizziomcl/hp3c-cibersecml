import os
from pathlib import Path
from dotenv import load_dotenv
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

# Carga las variables de entorno una sola vez al importar el módulo. Si no
# existe el archivo .env, se aplican los valores por defecto definidos en
# get_env().
load_dotenv()

# Paralelismo del entrenamiento (intra-modelo): número de árboles del Random
# Forest construidos en paralelo (scikit-learn n_jobs).
DEFAULT_RF_N_JOBS = 16

# Paralelismo del benchmark de transformación: número de fragmentos-proceso en
# los que se divide el dataset para el escalado. Es la variable experimental
# del benchmark HPC (se barre en {1,4,8,16}); este es solo el valor por defecto
# de una corrida suelta.
DEFAULT_HPC_NUM_WORKERS = 4

# Nota: ambos valores se acotan a los núcleos realmente disponibles mediante
# src.utils.hpc.resolver_workers, de modo que nunca se sobre-suscribe la CPU en
# máquinas o runners de CI con menos núcleos.


def get_env(key: str, default):
    """Lee una variable de entorno y la convierte al tipo de `default`."""
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
    """Construye los objetos de configuración de cada fase del pipeline a
    partir de las variables de entorno, con valores por defecto seguros."""

    def __init__(self, root_dir: Path | None = None):
        self.root_dir = Path(root_dir or os.getcwd())

    def _abs(self, rel: str) -> Path:
        """Convierte una ruta relativa al directorio raíz en absoluta."""
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
            num_workers=get_env("HPC_NUM_WORKERS", DEFAULT_HPC_NUM_WORKERS),
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
            n_jobs=get_env("N_JOBS", DEFAULT_RF_N_JOBS),
            random_state=get_env("RANDOM_STATE", 42),
        )
