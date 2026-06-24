import os
import sys
from pathlib import Path

# Hace importable `src.*` cuando pytest se invoca desde la raíz del repo.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Los tests nunca deben escribir logs en la raíz del repositorio.
os.environ.setdefault("LOGS_DIR", str(ROOT / ".pytest_logs"))
