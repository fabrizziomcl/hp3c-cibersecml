# Antivirus Inteligente con PCA + Random Forest
Este proyecto implementa un antivirus inteligente de demostración, construido con Python, Streamlit, PCA y Random Forest. El objetivo es simular un flujo real de detección de malware en memoria utilizando Machine Learning.

## Descripción del Proyecto
La aplicación:
1. Selecciona registros aleatorios del dataset Obfuscated-MalMem2022.csv.  
2. Aplica el pipeline usado durante el entrenamiento:
   - Selección de features  
   - Escalado (StandardScaler / RobustScaler)  
   - Reducción de dimensionalidad con PCA (3 componentes)  
3. Envía las muestras transformadas al modelo Random Forest.  
4. Muestra:
   - Etiqueta real generada desde Category  
   - Predicción del modelo  
   - Probabilidad de malware  
   - Category original del dataset  
   - Registros originales escaneados  

## Regla de Etiquetado Real
La etiqueta real se genera desde la columna Category:
- Category == "Benign" → BENIGNO  
- Category != "Benign" → MALICIOSO  

## Estructura del Proyecto
├── antivirus_app_streamlit.py
├── pipeline.py
├── modelo_randomforest_final.pkl
├── preprocesamiento_pca.pkl
├── Obfuscated-MalMem2022.csv
├── Clasificador_Binario.ipynb
├── Clustering.ipynb
├── .gitignore
└── README.md


## Requisitos
Instalar dependencias:
```bash
pip install requirements.txt
```


## Cómo Ejecutar la Aplicación

Asegurar que los archivos .pkl y .csv estén en el mismo directorio. Luego ejecutar:

streamlit run antivirus_app.py


## La interfaz se abrirá en:

http://localhost:8501

## Funcionamiento Interno

Se toman N muestras aleatorias del dataset.

Se genera la etiqueta real usando Category.

Las muestras pasan por el pipeline (scalers + PCA).

Se predice con Random Forest.

Se muestran resultados y registros originales.
