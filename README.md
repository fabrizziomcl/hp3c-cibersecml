# Intelligent Antivirus: HPC and MLOps Pipeline

Este repositorio implementa una infraestructura avanzada para la deteccion de malware mediante tecnicas de Machine Learning, optimizada con High Performance Computing (HPC) y gestionada bajo un ciclo de vida MLOps robusto. El sistema permite el entrenamiento paralelo de modelos, automatizacion de despliegues y monitoreo de rendimiento de hardware.

## Arquitectura del Sistema

La estructura esta diseñada para separar la experimentacion de la produccion, garantizando la escalabilidad y mantenibilidad:

- **dataset/**: Almacenamiento del dataset maestro original (fuente de verdad).
- **data/**: Almacenamiento de datasets en estados raw (particiones para entrenamiento), processed (cache numerica) y external (simulacion).
- **models/**: Repositorio de artefactos binarios (.pkl) listos para produccion.
- **notebooks/**: Entorno de investigacion y prototipado (Sandboxing).
- **src/**: Codigo fuente modular para ingesta, transformacion, entrenamiento y servicios API.
- **reports/**: Auditoria de experimentos, metricas de precision y reportes de eficiencia HPC.
- **docker/**: Configuracion de contenedores para garantizar paridad entre entornos.

## Flujo Operacional MLOps

### 1. Ingesta y Procesamiento HPC (Full Retraining)
El sistema utiliza procesamiento paralelo mediante `concurrent.futures` y `n_jobs=-1` en Scikit-Learn. El pipeline esta diseñado bajo una logica de **reentrenamiento total**: el componente de transformacion escanea y concatena automaticamente todos los archivos `.csv` presentes en `data/raw/`, eliminando duplicados y generando un modelo actualizado con la suma de toda la informacion historica disponible.

### 2. Automatizacion CI/CD
GitHub Actions orquestra el reentrenamiento continuo. El flujo se activa mediante:
- **Event-driven**: Nuevos datos detectados en `data/raw/`.
- **Manual**: Trigger directo desde la pestaña Actions para actualizaciones programadas.
El workflow ejecuta las pruebas, entrena el modelo en la nube y exporta los artefactos resultantes.

### 3. Contenerizacion y Despliegue
Mediante Docker Compose, el proyecto levanta una infraestructura dual:
- **API Backend**: Servicio FastAPI que expone el modelo para predicciones en tiempo real.
- **Frontend App**: Interfaz Streamlit para interaccion de usuario y visualizacion de resultados.

---

## Guia de Inicio Rapido

### Paso 1: Clonacion y Configuracion Inicial
Apenas se clone el repositorio, es necesario preparar el entorno:

1. Crear el archivo de variables de entorno:
   ```bash
   cp .env.example .env
   ```
2. Instalar las dependencias base (se recomienda usar un entorno virtual):
   ```bash
   pip install -r requirements.txt
   ```

### Paso 2: Fase de Experimentacion (Notebooks)
Para validar la logica de los datos y el rendimiento del modelo antes de automatizar:
- Ejecutar `notebooks/1_clasificador.ipynb`.
- Esto generara el primer `models/model.pkl` y los reportes iniciales.

### Paso 3: Fase de Produccion (Docker)
Para desplegar la arquitectura completa de servicios:
```bash
docker-compose up --build
```
- API disponible en `http://localhost:8000`
- Dashboard disponible en `http://localhost:8501`

---

## Casos de Uso y Valor Agregado

1. **Deteccion Adaptativa**: El sistema puede integrarse en redes corporativas para recibir flujos constantes de logs de memoria y actualizar su capacidad de deteccion sin intervencion humana manual.
2. **Benchmarking de Infraestructura**: Gracias a las metricas HPC (Speedup y Eficiencia), el proyecto sirve para auditar el rendimiento de clusters o instancias cloud destinadas a tareas de ciberseguridad.
3. **Auditoria Forense**: El registro historico en la carpeta `reports/` permite realizar trazabilidad de cuando y con que datos se entreno cada version del modelo, cumpliendo con estandares de cumplimiento y seguridad.
