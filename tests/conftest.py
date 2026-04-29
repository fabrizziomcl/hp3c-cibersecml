import os
import sys
from pathlib import Path

# Make `src.*` importable when pytest is invoked from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Tests must never write logs into the repo root.
os.environ.setdefault("LOGS_DIR", str(ROOT / ".pytest_logs"))
