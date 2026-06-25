# Antivirus Inteligente: Pipeline HPC y MLOps

Infraestructura para la detección de malware ofuscado mediante Machine Learning, con procesamiento paralelo (HPC) y un ciclo de vida MLOps automatizado: reentrenamiento continuo, promoción condicionada por métricas y despliegue contenerizado. El dataset base es CICMalMem-2022.

## Arquitectura

- **dataset/**: dataset maestro original (fuente de verdad).
- **data/**: particiones en `raw/` (entrenamiento), `processed/` (caché numérica) y `external/` (simulación de nuevos datos).
- **models/**: artefactos entrenados (`model.pkl`, `preprocessor.pkl`).
- **src/**: código fuente modular (ingesta, transformación, entrenamiento, inferencia y API).
- **reports/**: reportes versionados por corrida (`report.json` + gráficos).
- **tests/**: pruebas unitarias ejecutadas en CI.
- **notebooks/**: prototipado exploratorio.
- **Dockerfile / docker-compose.yml**: contenedores para paridad entre entornos.

## Flujo MLOps

### 1. Procesamiento paralelo (reentrenamiento total)
La transformación concatena automáticamente todos los `.csv` de `data/raw/`, deduplica y descarta columnas de varianza cero, de modo que cada corrida incorpora todo el historial disponible. El paralelismo se aplica en dos ejes independientes:

- **Escalado** (`HPC_NUM_WORKERS`): la matriz se divide en fragmentos que se escalan con `joblib.Parallel` (backend `loky`).
- **Entrenamiento del Random Forest** (`N_JOBS`): los árboles —y los pliegues de la validación cruzada— se construyen en paralelo vía `n_jobs` de scikit-learn.

Ambos valores se acotan automáticamente a los núcleos disponibles (`src/utils/hpc.py`), por lo que un valor de 16 nunca sobre-suscribe la CPU en máquinas o *runners* con menos núcleos. Cada corrida persiste un `report.json` con métricas del modelo, validación cruzada y rendimiento HPC (tiempos secuencial/paralelo, *speedup*, eficiencia y desglose de *workers*).

### 2. CI/CD con GitHub Actions
El workflow se activa por *push* a `data/raw/**` o `src/**`, o manualmente. Ejecuta los tests, reentrena en la nube y promueve el modelo solo si el *recall* no regresiona respecto a la corrida anterior (`src/utils/model_check.py`).

### 3. Servicios contenerizados
`docker-compose` levanta dos servicios desde la misma imagen:
- **API** (FastAPI/Uvicorn) en el puerto 8000.
- **Frontend** (Streamlit) en el puerto 8501.

## Inicio rápido

```bash
cp .env.example .env
pip install -r requirements.txt

python -m src.data.ingestion   # genera train_eval.csv y new_data_simulation.csv (split 80/20)
python -m src.models.train      # transforma, entrena y persiste model.pkl + preprocessor.pkl
pytest -q
```

### Aplicación
```bash
streamlit run antivirus_app.py            # UI local en http://localhost:8501
```

### Despliegue completo (Docker)
```bash
docker compose up --build
```
- API: `http://localhost:8000` (Swagger en `/docs`)
- Dashboard: `http://localhost:8501`

### Despliegue en Streamlit Community Cloud
Apunta a `antivirus_app.py` y selecciona **Python 3.12** en *Advanced settings* (para que los pines de `requirements.txt` instalen por *wheel*). El modelo viaja en el repositorio (`models/model.pkl`) y se carga por ruta relativa; no requiere *secrets*.

## Configuración

Variables principales en `.env` (ver `.env.example`):

| Variable | Por defecto | Descripción |
|---|---|---|
| `N_JOBS` | 16 | Paralelismo intra-modelo del Random Forest |
| `HPC_NUM_WORKERS` | 4 | Fragmentos-proceso del escalado |
| `RF_N_ESTIMATORS` / `RF_MAX_DEPTH` | 100 / 12 | Hiperparámetros del bosque |
| `PCA_COMPONENTS` | 3 | Componentes principales |
| `FORCE_IMBALANCE` | false | Submuestreo de malware (solo experimentación) |
| `RANDOM_STATE` | 42 | Semilla global |
