import json
import sys
from pathlib import Path
from typing import Optional

# Versión de esquema requerida para que un reporte sea comparable.
REQUIRED_SCHEMA_VERSION = 2


def _cargar_reporte(path: Path) -> Optional[dict]:
    """Carga y valida un report.json; devuelve None si no es comparable."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("schema_version") != REQUIRED_SCHEMA_VERSION:
        return None
    if "model_performance" not in data or "test" not in data["model_performance"]:
        return None
    if "recall" not in data["model_performance"]["test"]:
        return None
    return data


def _listar_corridas(reports_dir: Path = Path("reports")) -> list[Path]:
    """Lista los directorios de corridas (run_*) en orden cronológico inverso."""
    if not reports_dir.exists():
        return []
    return sorted(
        (d for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith("run_")),
        reverse=True,
    )


def check_improvement(reports_dir: Path = Path("reports")) -> bool:
    """Aprueba la promoción del modelo si el recall no presenta regresión.

    Compara el recall de la corrida más reciente contra el de la corrida válida
    anterior. Aprueba por defecto cuando no hay corridas previas comparables.
    """
    corridas = _listar_corridas(reports_dir)
    if not corridas:
        print("INFO: No se encontraron corridas. Se aprueba por defecto.")
        return True

    reporte_nuevo = _cargar_reporte(corridas[0] / "report.json")
    if reporte_nuevo is None:
        print(
            f"ERROR: El reporte más reciente en {corridas[0]} falta o tiene "
            "un esquema inválido."
        )
        return False

    reporte_anterior = next(
        (cargado for d in corridas[1:] if (cargado := _cargar_reporte(d / "report.json"))),
        None,
    )
    if reporte_anterior is None:
        print("INFO: No hay reporte previo compatible. Se aprueba por defecto.")
        return True

    recall_nuevo = reporte_nuevo["model_performance"]["test"]["recall"]
    recall_anterior = reporte_anterior["model_performance"]["test"]["recall"]
    print(
        f"DEBUG: recall_nuevo={recall_nuevo:.4f} | "
        f"recall_anterior={recall_anterior:.4f}"
    )

    if recall_nuevo >= recall_anterior:
        print("SUCCESS: El recall no presenta regresión. Se promueve el modelo.")
        return True
    print("WARNING: Regresión de recall detectada. Se omite la promoción.")
    return False


if __name__ == "__main__":
    sys.exit(0 if check_improvement() else 1)
