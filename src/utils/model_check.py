import os
import json
import sys
from pathlib import Path

def get_latest_reports():
    reports_dir = Path("reports")
    # Obtener todas las carpetas que empiezan con 'run_' y ordenarlas por nombre (fecha)
    runs = sorted([d for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith("run_")], reverse=True)
    return runs

def check_improvement():
    runs = get_latest_reports()
    
    if len(runs) < 2:
        print("INFO: Primera ejecucion detectada o no hay reportes previos. Aprobando por defecto.")
        return True

    new_report_path = runs[0] / "report.json"
    old_report_path = runs[1] / "report.json"

    try:
        with open(new_report_path, 'r') as f:
            new_data = json.load(f)
        with open(old_report_path, 'r') as f:
            old_data = json.load(f)

        new_recall = new_data["model_performance"]["test"]["recall"]
        old_recall = old_data["model_performance"]["test"]["recall"]

        print(f"DEBUG: Nuevo Recall = {new_recall:.4f}")
        print(f"DEBUG: Recall Anterior = {old_recall:.4f}")

        if new_recall >= old_recall:
            print("SUCCESS: El modelo ha mejorado o es igual. Procediendo al push.")
            return True
        else:
            print("WARNING: El modelo no superó el recall anterior. Se descarta la actualizacion.")
            return False
            
    except Exception as e:
        print(f"ERROR: Error al leer los reportes: {e}")
        return False

if __name__ == "__main__":
    if check_improvement():
        sys.exit(0)
    else:
        sys.exit(1)
