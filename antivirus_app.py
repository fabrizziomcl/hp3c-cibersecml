import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from src.pipelines.inference_pipeline import InferencePipeline
from src.utils.logger import logging

# Add project root to path for modular imports
sys.path.append(os.getcwd())

st.set_page_config(page_title="Antivirus Inteligente - CibersecML", layout="wide")

def main():
    st.title("🛡️ Antivirus Inteligente con HPC & Cloud MLOps")
    st.markdown("""
    Esta aplicación utiliza un modelo de Machine Learning (Random Forest) entrenado mediante un pipeline paralelo (HPC) 
    para detectar malware en memoria. 
    """)

    # Sidebar info
    st.sidebar.header("Configuración de Inferencia")
    # Updated default path to the partitioned dataset
    data_path = st.sidebar.text_input("Ruta del Dataset", "data/raw/train_eval.csv")

    if not os.path.exists(data_path):
        st.warning(f"Archivo no encontrado en {data_path}. Usando dataset original como fallback.")
        data_path = "dataset/Obfuscated-MalMem2022.csv"
        if not os.path.exists(data_path):
             st.error("No se encontró ningún dataset. Por favor ejecuta 'python src/data/ingestion.py' primero.")
             return

    # Load data
    df = pd.read_csv(data_path)
    
    st.subheader("Simulación de Escaneo de Memoria")
    num_samples = st.slider("Número de muestras a escanear", 1, 100, 10)
    
    if st.button("🚀 Iniciar Escaneo"):
        # Select random samples
        samples = df.sample(num_samples)
        
        # Prepare for inference (exclude target and category)
        features_to_scan = samples.drop(columns=['Class', 'Category'], errors='ignore')
        
        # Run inference pipeline
        pipeline = InferencePipeline()
        predictions = pipeline.predict(features_to_scan)
        
        # Display results
        results = samples.copy()
        results['Predicción'] = ['MALICIOSO' if p == 1 else 'BENIGNO' for p in predictions]
        
        # Real Label logic for comparison
        results['Etiqueta Real'] = results['Category'].apply(lambda x: 'MALICIOSO' if x != 'Benign' else 'BENIGNO')
        
        st.table(results[['Category', 'Etiqueta Real', 'Predicción']].head(20))
        
        # Metrics
        malware_count = (predictions == 1).sum()
        st.info(f"Escaneo finalizado. Detectados {malware_count} posibles hilos maliciosos de {num_samples} analizados.")

if __name__ == "__main__":
    main()
