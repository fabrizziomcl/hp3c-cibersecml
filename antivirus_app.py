# antivirus_app_streamlit.py
# -----------------------------------------------------------
# Antivirus inteligente basado en PCA + Random Forest.
# Ahora, la etiqueta real proviene de la columna "Category":
#   - Si Category == "Benign"  → Etiqueta real = BENIGNO
#   - Si Category != "Benign" → Etiqueta real = MALICIOSO
# -----------------------------------------------------------

import numpy as np
import pandas as pd
import streamlit as st
from pipeline import predecir


# ===========================================================
# 1. Cargar dataset
# ===========================================================

DATA_PATH = "Obfuscated-MalMem2022.csv"
CATEGORY_COL = "Category"     # Se usa esta columna para crear la etiqueta real


@st.cache_data
def load_dataset(path: str):
    """
    Carga el dataset y valida que exista la columna Category.
    """
    df = pd.read_csv(path)

    if CATEGORY_COL not in df.columns:
        raise ValueError(
            f"El dataset no contiene la columna '{CATEGORY_COL}', "
            "que es necesaria para generar la etiqueta real."
        )

    return df


df = load_dataset(DATA_PATH)


# ===========================================================
# 2. Función de escaneo
# ===========================================================

def escanear_n_registros(n: int):
    """
    Toma n registros aleatorios y genera:
        - Etiqueta real basada en Category
        - Predicción del modelo
        - Probabilidad de malware
    Devuelve:
        - DataFrame con resultados
        - Registros originales (para visualización)
    """
    muestras = df.sample(n)

    # Etiqueta real basada en Category
    categorias = muestras[CATEGORY_COL].astype(str).to_numpy()
    etiqueta_real = np.where(
        categorias == "Benign",
        "BENIGNO",
        "MALICIOSO"
    )

    # Predicción del modelo
    y_pred, y_proba = predecir(muestras)
    pred_txt = np.where(y_pred == 1, "MALICIOSO", "BENIGNO")

    # Tabla final de resultados
    resultados = pd.DataFrame({
        "Etiqueta REAL basada en Category": etiqueta_real,
        "Category (origen)": categorias,
        "Predicción modelo": pred_txt,
        "Probabilidad MALICIOSO": y_proba
    })

    return resultados, muestras


# ===========================================================
# 3. Interfaz Streamlit
# ===========================================================

st.set_page_config(
    page_title="Antivirus Inteligente",
    page_icon="",
    layout="centered",
)

st.title("Antivirus Inteligente - PCA + Random Forest")
st.write(
    """
    Esta aplicación simula un antivirus inteligente que analiza muestras del
    dataset de memoria maliciosa y predice si son benignas o maliciosas.
    
    La etiqueta real se obtiene a partir del campo **Category**:
    - Category = "Benign" → BENIGNO  
    - Category distinto → MALICIOSO  
    """
)

st.sidebar.header("Configuración del escaneo")
st.sidebar.write(f"Dataset cargado: {DATA_PATH}")
st.sidebar.write(f"Total de registros disponibles: {len(df)}")
st.sidebar.write(f"Columna usada como etiqueta real: Category")

st.sidebar.markdown("---")

n_registros = st.sidebar.slider(
    "Cantidad de registros a escanear:",
    min_value=1,
    max_value=50,
    value=1
)

scan_button = st.sidebar.button("Iniciar escaneo")


# ===========================================================
# 4. Ejecución principal
# ===========================================================

if scan_button:
    with st.spinner("Analizando registros..."):
        resultados, muestras = escanear_n_registros(n_registros)

    st.subheader("Resultado del último registro analizado")

    primera = resultados.iloc[0]

    st.markdown(
        f"""
        - Etiqueta REAL: **{primera['Etiqueta REAL basada en Category']}**  
        - Category original: **{primera['Category (origen)']}**  
        - Predicción del modelo: **{primera['Predicción modelo']}**  
        - Probabilidad de MALICIOSO: **{primera['Probabilidad MALICIOSO']:.4f}**
        """
    )

    st.markdown("---")

    st.write("Resultados completos del escaneo:")
    st.dataframe(
        resultados.style.format({"Probabilidad MALICIOSO": "{:.4f}"}),
        use_container_width=True
    )

    with st.expander("Ver registros originales del dataset"):
        st.dataframe(muestras.reset_index(drop=True), use_container_width=True)

else:
    st.info("Seleccione la cantidad de registros y presione 'Iniciar escaneo'.")
