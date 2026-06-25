"""Genera las figuras del barrido HPC a partir de resultados_hpc.json.

Script auxiliar FUERA del scope de la metodología. Produce figuras reales
(matplotlib, sin IA) de *speedup* y eficiencia frente al número de workers para
los dos experimentos, y las escribe en ``docs/figuras/`` para el informe.

Uso (desde la raíz del repo):
    PYTHONPATH=. python experimentos/generar_figuras.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RUTA_RESULTADOS = Path("experimentos/resultados_hpc.json")
DIR_FIGURAS = Path("docs/figuras")

COLOR_A = "#1f77b4"  # escalado (transformación)
COLOR_B = "#d62728"  # entrenamiento (Random Forest)
COLOR_IDEAL = "#7f7f7f"  # referencia ideal


def _etiquetas_y_valores(resultado: dict):
    """Extrae las etiquetas de workers solicitados y las series del resultado."""
    puntos = resultado["puntos"]
    etiquetas = [p["solicitados"] for p in puntos]
    speedups = [p["speedup"] for p in puntos]
    eficiencias = [p["eficiencia"] for p in puntos]
    return etiquetas, speedups, eficiencias


def figura_speedup(datos: dict, ruta_salida: Path):
    """Speedup vs número de workers para ambos experimentos + línea ideal."""
    xa, sa, _ = _etiquetas_y_valores(datos["experimento_a_escalado"])
    xb, sb, _ = _etiquetas_y_valores(datos["experimento_b_entrenamiento"])

    plt.figure(figsize=(7, 5))
    # Línea ideal (speedup lineal = N workers).
    plt.plot(xa, xa, "--", color=COLOR_IDEAL, label="Ideal (lineal)")
    plt.plot(xa, sa, "o-", color=COLOR_A, label="Escalado (transformación)")
    plt.plot(xb, sb, "s-", color=COLOR_B, label="Entrenamiento (Random Forest)")
    plt.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)

    plt.xticks(xa)
    plt.xlabel("Número de workers")
    plt.ylabel("Speedup ($T_{seq}/T_{par}$)")
    plt.title("Speedup frente al número de workers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    print(f"Figura guardada: {ruta_salida}")


def figura_eficiencia(datos: dict, ruta_salida: Path):
    """Eficiencia vs número de workers para ambos experimentos + ideal=1."""
    xa, _, ea = _etiquetas_y_valores(datos["experimento_a_escalado"])
    xb, _, eb = _etiquetas_y_valores(datos["experimento_b_entrenamiento"])

    plt.figure(figsize=(7, 5))
    plt.axhline(1.0, color=COLOR_IDEAL, linestyle="--", label="Ideal (eficiencia = 1)")
    plt.plot(xa, ea, "o-", color=COLOR_A, label="Escalado (transformación)")
    plt.plot(xb, eb, "s-", color=COLOR_B, label="Entrenamiento (Random Forest)")

    plt.xticks(xa)
    plt.xlabel("Número de workers")
    plt.ylabel("Eficiencia ($Speedup/N$)")
    plt.title("Eficiencia frente al número de workers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    print(f"Figura guardada: {ruta_salida}")


def main():
    if not RUTA_RESULTADOS.exists():
        raise SystemExit(
            f"No existe {RUTA_RESULTADOS}. Ejecuta primero benchmark_hpc.py."
        )
    with open(RUTA_RESULTADOS, "r", encoding="utf-8") as f:
        datos = json.load(f)

    DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    figura_speedup(datos, DIR_FIGURAS / "hpc_speedup_sweep.png")
    figura_eficiencia(datos, DIR_FIGURAS / "hpc_efficiency_sweep.png")


if __name__ == "__main__":
    main()
