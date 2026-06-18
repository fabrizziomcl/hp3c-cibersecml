import logging
import os
import sys
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
        format=_LOG_FORMAT,
        level=getattr(logging, log_level, logging.INFO),
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    logging.getLogger(__name__).info("Logging to %s", log_file)
    _configured = True


_configure_root_logger()
