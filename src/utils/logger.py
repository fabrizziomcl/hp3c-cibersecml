import logging
import os
from datetime import datetime
from pathlib import Path

_FORMATO_LOG = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
_configurado = False


def _configurar_logger_raiz() -> None:
    """Configura el logger raíz una sola vez. Importaciones posteriores son no-op."""
    global _configurado
    if _configurado:
        return

    logs_dir = Path(os.getenv("LOGS_DIR", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    archivo_log = logs_dir / f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    nivel_log = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        filename=str(archivo_log),
        format=_FORMATO_LOG,
        level=getattr(logging, nivel_log, logging.INFO),
        force=True,
    )
    _configurado = True


_configurar_logger_raiz()
