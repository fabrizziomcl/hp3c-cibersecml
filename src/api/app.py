from contextlib import asynccontextmanager
from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.pipelines.inference_pipeline import InferencePipeline
from src.utils.logger import logging

# Pipeline a nivel de módulo: se construye una sola vez y se reutiliza en todas
# las peticiones.
_pipeline = InferencePipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al arrancar la aplicación (evento de ciclo de vida)."""
    try:
        _pipeline.load()
        logging.info("Pipeline de inferencia listo")
    except Exception as e:
        logging.error("No se pudo cargar el modelo al arrancar: %s", e)
    yield


app = FastAPI(title="CybersecML - API de Antivirus Inteligente", lifespan=lifespan)


class PredictionRequest(BaseModel):
    """Cuerpo de una petición de predicción: vector de características."""

    features: List[float]


@app.get("/")
def read_root():
    """Endpoint raíz de verificación de disponibilidad."""
    return {"message": "Bienvenido a la API de Antivirus Inteligente"}


@app.get("/health")
def health():
    """Endpoint de estado: indica si el modelo está cargado."""
    return {"status": "ok", "model_loaded": _pipeline._model is not None}


@app.post("/predict")
def predict(data: List[PredictionRequest]):
    """Predice a partir de una lista de vectores de características en JSON."""
    if not data:
        raise HTTPException(status_code=400, detail="Carga vacía")
    df = pd.DataFrame([item.features for item in data])
    try:
        predictions = _pipeline.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inferencia fallida: {e}")
    return {"predictions": predictions.tolist()}


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """Predice a partir de un archivo CSV subido."""
    try:
        df = pd.read_csv(file.file)
        predictions = _pipeline.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inferencia fallida: {e}")
    return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
