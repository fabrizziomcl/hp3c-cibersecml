"""Interfaz Streamlit del antivirus: simula el escaneo de un instalador, muestra
el veredicto del modelo y permite aportar las muestras analizadas al dataset de
reentrenamiento. Pensada para desplegarse en Streamlit Community Cloud.

El reentrenamiento NO ocurre en la app: aportar datos crea un commit en
``data/raw/`` (vía la API de GitHub) que dispara el workflow CI/CD.
"""

import base64
import json
import random
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src.pipelines.inference_pipeline import InferencePipeline

st.set_page_config(page_title="CibersecML Antivirus", page_icon="🛡️", layout="centered")

SIMULATION_PATH = Path("data/external/new_data_simulation.csv")
REPORTS_DIR = Path("reports")
RAW_DIR = Path("data/raw")
DEFAULT_REPO = "fabrizziomcl/hp3c-cibersecml"
NO_FEATURE_COLS = ["Class", "Category", "row_id", "date_added", "source", "status"]

FAKE_APPS = [
    "setup_PhotoEditor_v3.1.exe",
    "installer_GameBooster_Pro.exe",
    "VPN_client_setup.exe",
    "driver_updater_v2.0.exe",
    "free_pdf_converter.exe",
    "media_player_setup.exe",
    "system_cleaner_pro.exe",
    "screen_recorder_installer.exe",
]

CSS = """
<style>
.title-block { text-align: center; padding: 1rem 0 0.5rem 0; }
.title-block h1 { font-size: 2.4rem; margin-bottom: 0; }
.title-block p  { color: #888; margin-top: 0.2rem; }
.file-box {
    display: flex; align-items: center; gap: 1rem;
    background: #1e1e2e; border: 1px solid #333;
    border-radius: 10px; padding: 1rem 1.4rem; margin: 1.5rem 0;
}
.file-icon { font-size: 2rem; }
.file-name { font-size: 1rem; color: #cdd6f4; }
.file-size { font-size: 0.8rem; color: #666; }
.result-card { border-radius: 12px; padding: 1.4rem 1.8rem; margin-top: 1.2rem; text-align: center; }
.result-safe   { background: #1a2e1a; border: 2px solid #40a02b; }
.result-danger { background: #2e1a1a; border: 2px solid #e64553; }
.result-label  { font-size: 2rem; font-weight: 800; margin-bottom: 0.4rem; }
.result-safe   .result-label { color: #40a02b; }
.result-danger .result-label { color: #e64553; }
.result-proba  { font-size: 1.1rem; color: #cdd6f4; margin: 0.2rem 0; }
.result-meta   { font-size: 0.82rem; color: #888; margin-top: 0.8rem; }
.metric-row    { display: flex; justify-content: center; gap: 2rem; margin-top: 0.6rem; }
.metric-value  { font-size: 1.2rem; font-weight: 700; color: #cba6f7; }
.metric-label  { font-size: 0.75rem; color: #888; }
</style>
"""


@st.cache_resource
def get_pipeline() -> InferencePipeline:
    pipeline = InferencePipeline()
    pipeline.load()
    return pipeline


@st.cache_data
def get_simulation_data() -> pd.DataFrame | None:
    return pd.read_csv(SIMULATION_PATH) if SIMULATION_PATH.exists() else None


def get_model_metrics() -> dict:
    """Métricas de prueba de la corrida más reciente con reporte válido."""
    if not REPORTS_DIR.exists():
        return {}
    for run in sorted(REPORTS_DIR.glob("run_*"), reverse=True):
        report = run / "report.json"
        if report.exists():
            data = json.loads(report.read_text(encoding="utf-8"))
            return data.get("model_performance", {}).get("test", {})
    return {}


def _secret(key: str):
    try:
        return st.secrets.get(key)
    except Exception:
        return None


def aportar_al_repositorio(muestras: pd.DataFrame) -> tuple[str, str]:
    """Persiste las muestras como un nuevo lote en ``data/raw/``.

    Con ``GITHUB_TOKEN`` configurado en *secrets* hace un commit vía la API de
    GitHub (dispara el reentrenamiento del CI/CD); en local escribe el archivo.
    """
    nombre = f"app_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    csv_bytes = muestras.to_csv(index=False).encode("utf-8")

    token = _secret("GITHUB_TOKEN")
    if token:
        repo = _secret("GITHUB_REPO") or DEFAULT_REPO
        api = f"https://api.github.com/repos/{repo}/contents/data/raw/{nombre}"
        resp = requests.put(
            api,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
            json={
                "message": f"data: aporte de {len(muestras)} muestras desde la app",
                "content": base64.b64encode(csv_bytes).decode(),
                "branch": _secret("GITHUB_BRANCH") or "main",
            },
            timeout=30,
        )
        if resp.status_code in (200, 201):
            return "cloud", f"data/raw/{nombre}"
        return "error", f"GitHub respondió {resp.status_code}: {resp.text[:200]}"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / nombre).write_bytes(csv_bytes)
    return "local", str(RAW_DIR / nombre)


def _fake_file_size() -> str:
    return f"{round(random.uniform(8.5, 120.0), 1)} MB"


def _escanear(simulacion: pd.DataFrame):
    """Anima el escaneo y devuelve (muestra, predicción, P(malware))."""
    estado = st.empty()
    barra = st.progress(0)
    pasos = [
        (15, "Verificando firma digital..."),
        (40, "Escaneando patrones en memoria..."),
        (70, "Consultando modelo de ML..."),
        (100, "Análisis completado."),
    ]
    for pct, msg in pasos:
        estado.markdown(f"🔍 **{msg}**")
        barra.progress(pct)
        time.sleep(0.4)
    barra.empty()
    estado.empty()

    muestra = simulacion.sample(1).reset_index(drop=True)
    features = muestra.drop(columns=NO_FEATURE_COLS, errors="ignore")
    pipeline = get_pipeline()
    return muestra, int(pipeline.predict(features)[0]), float(pipeline.predict_proba(features)[0][1])


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        "<div class='title-block'><h1>🛡️ CibersecML Antivirus</h1>"
        "<p>Detección de malware en memoria con Machine Learning</p></div>",
        unsafe_allow_html=True,
    )

    simulacion = get_simulation_data()
    if simulacion is None:
        st.error(
            "No se encontró el dataset de simulación. Ejecuta "
            "`python -m src.data.ingestion` o despliega con los datos versionados."
        )
        return

    st.session_state.setdefault("app_name", random.choice(FAKE_APPS))
    st.session_state.setdefault("app_size", _fake_file_size())
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("aportes", [])

    if st.button("🔀 Otro archivo"):
        st.session_state.app_name = random.choice(FAKE_APPS)
        st.session_state.app_size = _fake_file_size()
        st.session_state.result = None
        st.rerun()

    st.markdown(
        f"""
        <div class="file-box">
            <div class="file-icon">📦</div>
            <div>
                <div class="file-name">{st.session_state.app_name}</div>
                <div class="file-size">Tamaño: {st.session_state.app_size}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("⬇️ Instalar", type="primary", use_container_width=True):
        muestra, pred, proba = _escanear(simulacion)
        st.session_state.result = {
            "prediction": pred,
            "proba": proba,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.session_state.aportes.append(muestra)

    if st.session_state.result:
        r = st.session_state.result
        es_malware = r["prediction"] == 1
        metrics = get_model_metrics()
        recall = f"{metrics['recall'] * 100:.2f}%" if metrics.get("recall") else "N/D"
        acc = f"{metrics['accuracy'] * 100:.2f}%" if metrics.get("accuracy") else "N/D"
        st.markdown(
            f"""
            <div class="result-card {'result-danger' if es_malware else 'result-safe'}">
                <div class="result-label">{'⛔ MALICIOSO' if es_malware else '✅ BENIGNO'}</div>
                <div class="result-proba">Probabilidad de malware: <strong>{r['proba'] * 100:.1f}%</strong></div>
                <div class="metric-row">
                    <div><div class="metric-value">{acc}</div><div class="metric-label">Accuracy</div></div>
                    <div><div class="metric-value">{recall}</div><div class="metric-label">Recall</div></div>
                </div>
                <div class="result-meta">🕐 {r['timestamp']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if es_malware:
            st.error(f"⚠️ Se recomienda **no instalar** {st.session_state.app_name}.")
        else:
            st.success(f"✔️ {st.session_state.app_name} parece seguro.")

    st.divider()
    st.subheader("Aportar al reentrenamiento")
    aportes = st.session_state.aportes
    st.caption(
        f"{len(aportes)} muestra(s) analizada(s) en esta sesión. Al enviarlas se "
        "crea un lote en `data/raw/` que el pipeline CI/CD incorporará al reentrenar."
    )
    if aportes and st.button("📤 Enviar muestras al dataset", use_container_width=True):
        destino, detalle = aportar_al_repositorio(pd.concat(aportes, ignore_index=True))
        if destino == "cloud":
            st.success(f"Commit creado: `{detalle}`. El workflow se reentrenará.")
            st.session_state.aportes = []
        elif destino == "local":
            st.success(f"Guardado localmente en `{detalle}`.")
            st.session_state.aportes = []
        else:
            st.error(f"No se pudo subir: {detalle}")


if __name__ == "__main__":
    main()
