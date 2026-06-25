# Guía para recrear los experimentos

Secuencia verificada para reproducir, desde cero, todos los experimentos del proyecto: las dos corridas de clasificación (desbalanceada y balanceada), el barrido de escalabilidad HPC y la validación en la VM de GitHub. Se ejecuta desde la raíz del repositorio.

## Antes de empezar: qué NO borrar
- **`dataset/Obfuscated-MalMem2022.csv`** — fuente de verdad; sin él la ingesta falla.
- **`experimentos/`** — scripts del barrido HPC (no versionados en git; viven solo en tu disco).

## 0. Estado limpio (opcional)
Para partir "en limpio" dejando las carpetas:
```powershell
Remove-Item -Recurse -Force reports\run_*
Remove-Item -Force data\raw\*, data\processed\*, data\external\* -ErrorAction SilentlyContinue
Remove-Item -Force models\*.pkl -ErrorAction SilentlyContinue
```

## 1. Entorno
```powershell
conda activate hp3c          # o python -m venv hp3c; .\hp3c\Scripts\Activate.ps1
pip install -r requirements.txt
```
Las dependencias fijadas tienen *wheels* para Python 3.10–3.12; en versiones más nuevas se compilan desde el código fuente.

## 2. Ingesta de datos
```powershell
python -m src.data.ingestion
```
Genera `data/raw/train_eval.csv` (80 %) y `data/external/new_data_simulation.csv` (20 %).

## 3. Reconstruir el dataset acumulado
El entrenamiento concatena todos los `.csv` de `data/raw/`. Para reproducir el dataset completo (~58 047 filas tras deduplicación), simula la llegada del lote nuevo:
```powershell
Copy-Item data\external\new_data_simulation.csv data\raw\new_data_simulation_1.csv
```

## 4. Corrida 1 — línea base desbalanceada
En `.env`: `FORCE_IMBALANCE=true`. Luego:
```powershell
python -m src.models.train
```
Crea `reports/run_<ts>/` con `report.json`, matriz de confusión y curva ROC.

## 5. Corrida 2 — modelo balanceado (producción)
En `.env`: `FORCE_IMBALANCE=false`. Luego:
```powershell
python -m src.models.train
```
`model_check` aprueba la promoción si el recall no regresiona respecto a la corrida anterior.

## 6. Barrido de escalabilidad HPC (1/4/8/16)
```powershell
$env:PYTHONPATH = "."
python experimentos\benchmark_hpc.py --reps 7
python experimentos\generar_figuras.py
```
Produce `experimentos/resultados_hpc.json` y las figuras `docs/figuras/hpc_speedup_sweep.png` y `hpc_efficiency_sweep.png`.

## 7. Validación en la VM de GitHub (comparación local vs CI)
```powershell
git add -A
git commit -m "chore: regenera dataset, modelos y reportes"
git push origin main
```
El workflow reentrena en el *runner* (4 vCPU), pasa el *gate* de recall y el bot publica su `report.json`. Recupéralo:
```powershell
git pull origin main
```
El `report.json` del run del bot contiene las métricas HPC medidas en la VM, base de la comparación local-vs-CI.

## 8. Pruebas
```powershell
$env:PYTHONPATH = "."
pytest -q
```

## Resultado verificado
La secuencia produce dos corridas locales (desbalanceada y balanceada) más la corrida de la VM, todas bajo el mismo esquema de reporte. El modelo alcanza recall ≈ 0.999 (AUC ≈ 1.0) y se mantiene robusto ante el desbalance forzado; el barrido confirma que el entrenamiento del Random Forest escala (speedup hasta ~7.9×) mientras que el escalado de la transformación está dominado por el overhead (speedup < 1).
