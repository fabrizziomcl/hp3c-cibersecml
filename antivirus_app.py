import os

import pandas as pd
import streamlit as st

from src.pipelines.inference_pipeline import InferencePipeline

st.set_page_config(page_title="Antivirus Inteligente - CibersecML", layout="wide")


@st.cache_resource
def get_pipeline() -> InferencePipeline:
    """Load model + preprocessor once for the entire Streamlit session."""
    pipeline = InferencePipeline()
    pipeline.load()
    return pipeline


def main():
    st.title("🛡️ Antivirus Inteligente con HPC & Cloud MLOps")
    st.markdown(
        "Modelo Random Forest entrenado con un pipeline paralelo para detectar "
        "malware en memoria sobre el dataset Obfuscated-MalMem2022."
    )

    st.sidebar.header("Configuración de Inferencia")
    data_path = st.sidebar.text_input("Ruta del Dataset", "data/raw/train_eval.csv")

    if not os.path.exists(data_path):
        st.warning(f"Archivo no encontrado en {data_path}. Usando dataset original como fallback.")
        data_path = "dataset/Obfuscated-MalMem2022.csv"
        if not os.path.exists(data_path):
            st.error("No se encontró ningún dataset. Ejecuta `python src/data/ingestion.py` primero.")
            return

    df = pd.read_csv(data_path)

    st.subheader("Simulación de Escaneo de Memoria")
    num_samples = st.slider("Número de muestras a escanear", 1, 100, 10)

    if st.button("🚀 Iniciar Escaneo"):
        samples = df.sample(num_samples)
        features = samples.drop(columns=["Class", "Category"], errors="ignore")

        pipeline = get_pipeline()
        predictions = pipeline.predict(features)
        probas = pipeline.predict_proba(features)[:, 1]

        results = samples.copy()
        results["Predicción"] = ["MALICIOSO" if p == 1 else "BENIGNO" for p in predictions]
        results["P(malware)"] = probas.round(4)
        if "Category" in results.columns:
            results["Etiqueta Real"] = results["Category"].apply(
                lambda x: "MALICIOSO" if x != "Benign" else "BENIGNO"
            )
            display_cols = ["Category", "Etiqueta Real", "Predicción", "P(malware)"]
        else:
            display_cols = ["Predicción", "P(malware)"]

        st.table(results[display_cols].head(20))

        malware_count = int((predictions == 1).sum())
        st.info(
            f"Escaneo finalizado. Detectados {malware_count} hilos potencialmente "
            f"maliciosos de {num_samples} analizados."
        )


if __name__ == "__main__":
    main()
