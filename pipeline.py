import sys
import os
from src.pipelines.inference_pipeline import InferencePipeline

# Bridge to the new modular system for backward compatibility
def predecir(df_samples):
    """
    Función de compatibilidad para la app de Streamlit antigua o scripts de test.
    """
    pipeline = InferencePipeline()
    # Excluimos columnas que el modelo no espera si están presentes
    X = df_samples.drop(columns=['Category', 'Class'], errors='ignore')
    
    preds = pipeline.predict(X)
    
    # En el sistema antiguo se esperaba y_pred, y_proba
    # Como simplificamos el pipeline, devolveremos probas simuladas basadas en la predicción por ahora
    y_proba = [0.99 if p == 1 else 0.01 for p in preds] 
    
    return preds, y_proba

if __name__ == "__main__":
    import pandas as pd
    # Intenta cargar datos de la partición de entrenamiento
    path = "data/raw/train_eval.csv"
    if os.path.exists(path):
        df = pd.read_csv(path).head(5)
        p, prob = predecir(df)
        print(f"Predicciones (Top 5): {p}")
