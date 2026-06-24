import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.utils.exception import CustomException
from src.utils.logger import logging


class DataIngestion:
    """Parte el dataset maestro en dos particiones.

    Genera una partición de entrenamiento/evaluación (``data/raw/``) y una
    partición de simulación (``data/external/``) que alimenta el reentrenamiento
    disparado por el flujo CI/CD.
    """

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        """Carga el dataset maestro, lo divide y persiste ambas particiones."""
        logging.info("Iniciando la ingesta de datos")
        try:
            origen = self.config.local_data_file
            if not origen.exists():
                raise FileNotFoundError(f"Dataset maestro no encontrado en {origen}")

            df = pd.read_csv(origen)
            logging.info("Dataset maestro cargado: shape=%s", df.shape)

            os.makedirs(self.config.train_eval_path.parent, exist_ok=True)
            os.makedirs(self.config.simulation_path.parent, exist_ok=True)

            # Estratifica por la variable objetivo si está presente, para
            # conservar la proporción de clases en ambas particiones.
            estratificar = df["Class"] if "Class" in df.columns else None
            train_eval_df, simulation_df = train_test_split(
                df,
                test_size=self.config.simulation_split_size,
                random_state=self.config.random_state,
                stratify=estratificar,
            )

            train_eval_df.to_csv(self.config.train_eval_path, index=False)
            simulation_df.to_csv(self.config.simulation_path, index=False)

            logging.info(
                "Ingesta completada. train_eval=%d, simulación=%d",
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
