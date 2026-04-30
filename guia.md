# Guía para correr el proyecto

Esta es la secuencia que sí funcionó en este repositorio, ejecutada desde la raíz del proyecto `/workspaces/hp3c-cibersecml`.

## 1. Crear el entorno virtual

```bash
python3 -m venv hp3c
```

## 2. Activar el entorno

```bash
source hp3c/bin/activate
```

Al activarlo, la terminal debe mostrar algo como `(hp3c)` al inicio.

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 4. Ejecutar la ingesta de datos

Usa el comando como módulo para evitar problemas de importación:

```bash
python -m src.data.ingestion
```

Esto genera:

- `data/raw/train_eval.csv`
- `data/external/new_data_simulation.csv`

## 5. Entrenar el modelo

```bash
python -m src.models.train
```

Esto persiste:

- `models/model.pkl`
- `models/preprocessor.pkl`

## 6. Ejecutar pruebas

```bash
pytest -q
```

## Resultado verificado

Con esa secuencia, el proyecto quedó funcionando y las pruebas pasaron correctamente.
