# inferencia_pca_rf.py
import joblib
import pandas as pd
import numpy as np

# ==========================
# Cargar modelo y preproceso
# ==========================

modelo = joblib.load("modelo_randomforest_final.pkl")
preproc = joblib.load("preprocesamiento_pca.pkl")

FEATURE_COLS   = preproc["feature_cols"]
normal_like    = preproc["normal_like"]
non_normal     = preproc["non_normal"]
scalers        = preproc["scalers"]
pca_3          = preproc["pca_3"]

# Asumimos: scalers[0] = StandardScaler (normales)
#          scalers[1] = RobustScaler   (no normales)
ss = scalers[0]
rs = scalers[1]


def preprocess_df(df: pd.DataFrame) -> np.ndarray:
    """
    1) Se queda solo con FEATURE_COLS (si existen en df)
    2) Reconstruye las columnas que esperan los scalers (ss y rs)
       añadiendo columnas faltantes con 0.0
    3) Aplica los escaladores
    4) Aplica el PCA de 3 componentes
    Devuelve una matriz lista para el modelo.
    """
    # 1) seleccionar columnas de interés
    X = df.copy()

    # Nos quedamos solo con las columnas de features que guardaste
    cols_existentes = [c for c in FEATURE_COLS if c in X.columns]
    X = X[cols_existentes]

    # ---------- Escalado para columnas "normales" (ss) ----------
    if hasattr(ss, "feature_names_in_"):
        norm_expected = list(ss.feature_names_in_)
    else:
        norm_expected = [c for c in normal_like if c in X.columns]

    # Asegurar que todas las columnas esperadas existan en X
    for col in norm_expected:
        if col not in X.columns:
            X[col] = 0.0  # valor neutro

    # Ordenar las columnas exactamente como las espera el scaler
    X_norm = X[norm_expected]
    X_scaled_norm = ss.transform(X_norm)
    X.loc[:, norm_expected] = X_scaled_norm

    # ---------- Escalado para columnas "no normales" (rs) ----------
    if hasattr(rs, "feature_names_in_"):
        non_norm_expected = list(rs.feature_names_in_)
    else:
        non_norm_expected = [c for c in non_normal if c in X.columns]

    for col in non_norm_expected:
        if col not in X.columns:
            X[col] = 0.0

    X_non_norm = X[non_norm_expected]
    X_scaled_non_norm = rs.transform(X_non_norm)
    X.loc[:, non_norm_expected] = X_scaled_non_norm

    # ---------- PCA ----------
    # Si el PCA recuerda nombres de columnas, alineamos igual
    if hasattr(pca_3, "feature_names_in_"):
        pca_cols = list(pca_3.feature_names_in_)
        for col in pca_cols:
            if col not in X.columns:
                X[col] = 0.0
        X_for_pca = X[pca_cols]
    else:
        # si no tiene feature_names_in_, usamos todo X
        X_for_pca = X

    X_pca = pca_3.transform(X_for_pca)

    return X_pca



def predecir(df: pd.DataFrame):
    """
    df: DataFrame crudo (puede traer más columnas, incluso 'Class').
    Devuelve: (predicciones, prob_clase_1)
    """
    X_pca = preprocess_df(df)
    y_pred = modelo.predict(X_pca)
    y_proba = modelo.predict_proba(X_pca)[:, 1]
    return y_pred, y_proba


# Ejemplo de uso desde consola (puedes borrarlo en producción)
if __name__ == "__main__":
    df_nuevo = pd.read_csv("nuevos_casos.csv")   # <-- tu archivo de entrada
    preds, probas = predecir(df_nuevo)

    print("Predicciones:", preds[:10])
    print("Probabilidades clase 1:", probas[:10])
