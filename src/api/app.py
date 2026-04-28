from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import os
from pydantic import BaseModel
from typing import List
import uvicorn
from src.pipelines.inference_pipeline import InferencePipeline

app = FastAPI(title="CybersecML - Intelligent Antivirus API")

class PredictionRequest(BaseModel):
    features: List[float]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Intelligent Antivirus API"}

@app.post("/predict")
def predict(data: List[PredictionRequest]):
    # Convert request data to DataFrame
    df = pd.DataFrame([item.features for item in data])
    
    pipeline = InferencePipeline()
    predictions = pipeline.predict(df)
    
    return {"predictions": predictions.tolist()}

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    pipeline = InferencePipeline()
    predictions = pipeline.predict(df)
    
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
