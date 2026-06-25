"""Barrido de escalabilidad HPC (1/4/8/16 workers).

Script auxiliar FUERA del scope de la metodología de producción: mide el
*speedup* y la eficiencia de los dos ejes de paralelismo del proyecto para una
rejilla de números de workers, con varias repeticiones promediadas.

    Experimento A — Escalado de la transformación
        Divide la matriz de características en N fragmentos-proceso y los escala
        en paralelo con ``joblib`` (backend ``loky``). Paralelismo por
        partición de datos.

    Experimento B — Entrenamiento del Random Forest
        Entrena el clasificador con ``n_jobs`` ∈ rejilla. Paralelismo
        intra-modelo (árboles construidos en paralelo).

Reutiliza el código de producción (``src/``) para cargar y limpiar los datos
reales, de modo que el benchmark refleje exactamente la carga del pipeline.

Uso (desde la raíz del repo):
    PYTHONPATH=. python experimentos/benchmark_hpc.py [--reps 5]

Genera ``experimentos/resultados_hpc.json`` con los resultados crudos y
promediados, consumible por ``experimentos/generar_figuras.py``.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config.config import ConfigurationManager
from src.data.transformation import DataTransformation, _escalar_fragmento
from src.utils.hpc import detectar_nucleos_disponibles

# Rejilla de workers solicitada para el barrido.
REJILLA_WORKERS = [1, 4, 8, 16]

# Repeticiones por configuración (se promedian; la primera es de calentamiento).
REPETICIONES_POR_DEFECTO = 5

RUTA_RESULTADOS = Path("experimentos/resultados_hpc.json")


def _cronometrar(funcion, repeticiones: int) -> dict:
    """Ejecuta ``funcion`` con calentamiento y devuelve estadísticos robustos.

    La primera ejecución (calentamiento) no se contabiliza: amortiza costos
    únicos como la creación del pool de procesos de ``loky``. Se reporta la
    **mediana** como valor central por ser robusta a los picos esporádicos de
    carga del sistema operativo.
    """
    funcion()  # calentamiento
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        funcion()
        tiempos.append(time.perf_counter() - inicio)
    return {
        "mediana": statistics.median(tiempos),
        "media": statistics.mean(tiempos),
        "std": statistics.pstdev(tiempos) if len(tiempos) > 1 else 0.0,
        "min": min(tiempos),
        "tiempos": tiempos,
    }


def _construir_rejilla_efectiva(nucleos: int) -> list[dict]:
    """Mapea la rejilla solicitada a workers efectivos (acotados a núcleos).

    Conserva la etiqueta solicitada para el eje de las figuras, pero usa el
    valor efectivo para no sobre-suscribir la CPU.
    """
    rejilla = []
    for solicitados in REJILLA_WORKERS:
        efectivos = max(1, min(solicitados, nucleos))
        rejilla.append(
            {
                "solicitados": solicitados,
                "efectivos": efectivos,
                "recortado": solicitados > nucleos,
            }
        )
    return rejilla


def benchmark_escalado(X: np.ndarray, rejilla: list[dict], repeticiones: int) -> dict:
    """Experimento A: escalado secuencial vs paralelo por número de workers."""
    from joblib import Parallel, delayed

    scaler = StandardScaler().fit(X)
    media, escala = scaler.mean_, scaler.scale_

    # Referencia secuencial pura (sin joblib): escalado vectorizado de todo el
    # arreglo. Es el cómputo que realmente se ejecutaría sin paralelizar.
    base = _cronometrar(lambda: _escalar_fragmento(X, media, escala), repeticiones)
    t_seq = base["mediana"]
    print(f"[A] escalado secuencial puro: {t_seq*1000:.3f} ms")

    puntos = []
    for cfg in rejilla:
        n = cfg["efectivos"]

        def correr(n=n):
            fragmentos = np.array_split(X, n)
            partes = Parallel(n_jobs=n, backend="loky")(
                delayed(_escalar_fragmento)(c, media, escala) for c in fragmentos
            )
            return np.vstack(partes)

        medida = _cronometrar(correr, repeticiones)
        t_par = medida["mediana"]
        speedup = t_seq / t_par if t_par > 0 else float("inf")
        eficiencia = speedup / n
        puntos.append(
            {
                **cfg,
                "t_par_mediana": t_par,
                "t_par_std": medida["std"],
                "speedup": speedup,
                "eficiencia": eficiencia,
            }
        )
        print(
            f"[A] workers={cfg['solicitados']:>2} (efectivos={n:>2}): "
            f"t_par={t_par*1000:8.3f} ms  speedup={speedup:6.3f}x  "
            f"efic={eficiencia:6.3f}"
        )

    return {"t_seq": t_seq, "t_seq_std": base["std"], "puntos": puntos}


def benchmark_entrenamiento(
    X_train: np.ndarray,
    y_train: np.ndarray,
    rejilla: list[dict],
    repeticiones: int,
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> dict:
    """Experimento B: entrenamiento del RF por número de núcleos (n_jobs)."""

    def entrenar(n_jobs):
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        rf.fit(X_train, y_train)

    # Se cronometra cada punto de la rejilla en el mismo marco (sklearn n_jobs)
    # y el speedup se ancla al punto de un solo núcleo, que es la definición
    # estándar de speedup paralelo y evita inconsistencias entre mediciones.
    medidas = {}
    for cfg in rejilla:
        n = cfg["efectivos"]
        if n not in medidas:
            medidas[n] = _cronometrar(lambda n=n: entrenar(n), repeticiones)

    t_seq = medidas[1]["mediana"] if 1 in medidas else min(
        m["mediana"] for m in medidas.values()
    )
    print(f"[B] referencia n_jobs=1: {t_seq:.4f} s")

    puntos = []
    for cfg in rejilla:
        n = cfg["efectivos"]
        t_par = medidas[n]["mediana"]
        speedup = t_seq / t_par if t_par > 0 else float("inf")
        eficiencia = speedup / n
        puntos.append(
            {
                **cfg,
                "t_par_mediana": t_par,
                "t_par_std": medidas[n]["std"],
                "speedup": speedup,
                "eficiencia": eficiencia,
            }
        )
        print(
            f"[B] n_jobs={cfg['solicitados']:>2} (efectivos={n:>2}): "
            f"t_par={t_par:7.4f} s  speedup={speedup:6.3f}x  "
            f"efic={eficiencia:6.3f}"
        )

    return {"t_seq": t_seq, "t_seq_std": medidas.get(1, {}).get("std", 0.0), "puntos": puntos}


def main():
    parser = argparse.ArgumentParser(description="Barrido de escalabilidad HPC")
    parser.add_argument(
        "--reps",
        type=int,
        default=REPETICIONES_POR_DEFECTO,
        help="repeticiones por configuración (se promedian)",
    )
    args = parser.parse_args()

    nucleos = detectar_nucleos_disponibles()
    rejilla = _construir_rejilla_efectiva(nucleos)
    print(f"Núcleos disponibles: {nucleos} | repeticiones: {args.reps}")
    print(f"Rejilla solicitada: {REJILLA_WORKERS}\n")

    cfg = ConfigurationManager()
    dt_cfg = cfg.get_data_transformation_config()
    mt_cfg = cfg.get_model_trainer_config()

    # Carga y limpieza de datos reales reutilizando el pipeline de producción.
    transformador = DataTransformation(dt_cfg)
    df = transformador._cargar_y_limpiar()
    feature_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in ("Class", "Category")
    ]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["Class"].to_numpy()
    print(f"Datos: {X.shape[0]} filas x {X.shape[1]} features\n")

    # --- Experimento A: escalado ---
    print("=== Experimento A: escalado de la transformación ===")
    resultado_a = benchmark_escalado(X, rejilla, args.reps)

    # --- Datos transformados para el Experimento B ---
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline

    preprocesador = Pipeline(
        [("scaler", StandardScaler()), ("pca", PCA(n_components=dt_cfg.pca_components))]
    )
    X_pca = preprocesador.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(
        X_pca,
        y,
        test_size=mt_cfg.test_size,
        random_state=mt_cfg.random_state,
        stratify=y,
    )

    print("\n=== Experimento B: entrenamiento del Random Forest ===")
    resultado_b = benchmark_entrenamiento(
        X_train,
        y_train,
        rejilla,
        args.reps,
        mt_cfg.params_n_estimators,
        mt_cfg.params_max_depth,
        mt_cfg.random_state,
    )

    salida = {
        "config": {
            "nucleos_disponibles": nucleos,
            "rejilla_solicitada": REJILLA_WORKERS,
            "repeticiones": args.reps,
            "n_filas": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "rf_n_estimators": mt_cfg.params_n_estimators,
            "rf_max_depth": mt_cfg.params_max_depth,
        },
        "experimento_a_escalado": resultado_a,
        "experimento_b_entrenamiento": resultado_b,
    }
    RUTA_RESULTADOS.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
        json.dump(salida, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
