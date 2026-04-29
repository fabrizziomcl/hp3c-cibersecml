from contextlib import asynccontextmanager
from typing import List

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.pipelines.inference_pipeline import InferencePipeline
from src.utils.logger import logging

# Module-level pipeline so it is constructed once and reused across requests.
_pipeline = InferencePipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _pipeline.load()
        logging.info("Inference pipeline ready")
    except Exception as e:
        logging.error("Could not load model at startup: %s", e)
    yield


app = FastAPI(title="CybersecML - Intelligent Antivirus API", lifespan=lifespan)


class PredictionRequest(BaseModel):
    features: List[float]


@app.get("/")
def read_root():
    return {"message": "Welcome to the Intelligent Antivirus API"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _pipeline._model is not None}


@app.post("/predict")
def predict(data: List[PredictionRequest]):
    if not data:
        raise HTTPException(status_code=400, detail="Empty payload")
    df = pd.DataFrame([item.features for item in data])
    try:
        predictions = _pipeline.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    return {"predictions": predictions.tolist()}


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        predictions = _pipeline.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
