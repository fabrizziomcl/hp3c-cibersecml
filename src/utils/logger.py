import logging
import os
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
_configured = False


def _configure_root_logger() -> None:
    """Configure the root logger once. Subsequent imports are no-ops."""
    global _configured
    if _configured:
        return

    logs_dir = Path(os.getenv("LOGS_DIR", "logs"))
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        filename=str(log_file),
        format=_LOG_FORMAT,
        level=getattr(logging, log_level, logging.INFO),
        force=True,
    )
    _configured = True


_configure_root_logger()
