# model_monitoring.py
# Monitoreo de modelo y deteccion de Data Drift — Streamlit
# Ejecutar con:  streamlit run model_monitoring.py
# Requiere API corriendo:  uvicorn main:app --port 8000
## ..\venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
## ..\venv\Scripts\streamlit run model_monitoring.py



import os
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split

# Configuracion

API_URL = "http://localhost:8000"
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Base_de_datos.csv")
MONITOR_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitoring_log.csv")

NUMERIC_FEATURES = [
    "puntaje_datacredito", "huella_consulta",
    "capital_prestado", "plazo_meses", "cuota_pactada",
    "salario_cliente", "total_otros_prestamos",
    "promedio_ingresos_datacredito", "edad_cliente",
    "cant_creditosvigentes",
    "creditos_sectorFinanciero", "creditos_sectorCooperativo",
    "creditos_sectorReal",
]

CATEGORICAL_FEATURES = ["tipo_credito", "tipo_laboral"]
ORDINAL_FEATURES = ["tendencia_ingresos"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES

LEAKAGE_COLUMNS = [
    "puntaje", "saldo_mora", "saldo_total",
    "saldo_principal", "saldo_mora_codeudor",
]

TARGET = "Pago_atiempo"

# Carga y separacion de datos

@st.cache_data
def load_data():
    """Carga el CSV, elimina leakage, separa en referencia y nuevos datos."""
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=["fecha_prestamo"], errors="ignore")
    df = df.drop(columns=[c for c in LEAKAGE_COLUMNS if c in df.columns], errors="ignore")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_ref, X_new, y_ref, y_new = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return (
        X_ref.reset_index(drop=True),
        X_new.reset_index(drop=True),
        y_ref.reset_index(drop=True),
        y_new.reset_index(drop=True),
    )

# Consumo de API

def get_predictions(X_batch: pd.DataFrame) -> dict | None:
    """Envia batch de datos al endpoint /predict_batch y retorna resultado."""
    records = X_batch.to_dict(orient="records")

    # Limpiar NaN y convertir categoricas a string para JSON valido
    clean = []
    for rec in records:
        cr = {}
        for k, v in rec.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                cr[k] = None
            elif k in ("tipo_credito", "tipo_laboral", "tendencia_ingresos") and v is not None:
                cr[k] = str(v)
            else:
                cr[k] = v
        clean.append(cr)

    try:
        response = requests.post(f"{API_URL}/predict_batch", json=clean, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar a la API. Verifica que este corriendo en http://localhost:8000")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error al consultar la API: {e}")
        return None

# Logging de predicciones

def log_predictions(X_batch: pd.DataFrame, predictions: list, probabilities: list):
    """Guarda features + prediction + probability + timestamp en monitoring_log.csv."""
    df_log = X_batch.copy()
    df_log["prediction"] = predictions
    df_log["probability"] = probabilities
    df_log["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_path = Path(MONITOR_LOG)
    if log_path.exists():
        df_log.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_log.to_csv(log_path, index=False)

def load_log() -> pd.DataFrame:
    """Carga el log de monitoreo si existe."""
    if Path(MONITOR_LOG).exists():
        return pd.read_csv(MONITOR_LOG)
    return pd.DataFrame()

# Metricas de Data Drift

def ks_test(ref: pd.Series, cur: pd.Series) -> dict:
    """Kolmogorov-Smirnov test para variables numericas."""
    stat, p = stats.ks_2samp(ref.dropna(), cur.dropna())
    return {"metric": "KS", "statistic": round(stat, 6), "p_value": round(p, 6), "drift": p < 0.05}

def psi_metric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> dict:
    """Population Stability Index."""
    r, c = ref.dropna().values, cur.dropna().values
    breakpoints = np.percentile(r, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    r_pct = (np.histogram(r, bins=breakpoints)[0] + 1)
    c_pct = (np.histogram(c, bins=breakpoints)[0] + 1)
    r_pct, c_pct = r_pct / r_pct.sum(), c_pct / c_pct.sum()

    psi_val = float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))
    return {"metric": "PSI", "statistic": round(psi_val, 6), "p_value": None, "drift": psi_val > 0.2}

def js_divergence(ref: pd.Series, cur: pd.Series, bins: int = 30) -> dict:
    """Jensen-Shannon divergence."""
    r, c = ref.dropna().values, cur.dropna().values
    edges = np.histogram_bin_edges(np.concatenate([r, c]), bins=bins)
    r_hist = np.histogram(r, bins=edges)[0] + 1
    c_hist = np.histogram(c, bins=edges)[0] + 1
    js = float(jensenshannon(r_hist / r_hist.sum(), c_hist / c_hist.sum()) ** 2)
    return {"metric": "Jensen-Shannon", "statistic": round(js, 6), "p_value": None, "drift": js > 0.1}

def chi2_test(ref: pd.Series, cur: pd.Series) -> dict:
    """Chi-cuadrado para variables categoricas."""
    r, c = ref.dropna().astype(str), cur.dropna().astype(str)
    cats = sorted(set(r.unique()) | set(c.unique()))
    r_counts = r.value_counts().reindex(cats, fill_value=0)
    c_counts = c.value_counts().reindex(cats, fill_value=0)
    expected = (r_counts / r_counts.sum() * c_counts.sum()).replace(0, 1e-10)
    stat, p = stats.chisquare(c_counts.values, f_exp=expected.values)
    return {"metric": "Chi-Squared", "statistic": round(float(stat), 6), "p_value": round(float(p), 6), "drift": p < 0.05}

def compute_drift_report(df_ref: pd.DataFrame, df_cur: pd.DataFrame) -> pd.DataFrame:
    """Calcula todas las metricas de drift entre referencia y actual."""
    rows = []
    for col in NUMERIC_FEATURES:
        if col in df_ref.columns and col in df_cur.columns:
            for fn in (ks_test, psi_metric, js_divergence):
                result = fn(df_ref[col], df_cur[col])
                result["variable"] = col
                rows.append(result)

    for col in CATEGORICAL_FEATURES + ORDINAL_FEATURES:
        if col in df_ref.columns and col in df_cur.columns:
            result = chi2_test(df_ref[col], df_cur[col])
            result["variable"] = col
            rows.append(result)

    if not rows:
        return pd.DataFrame(columns=["variable", "metric", "statistic", "p_value", "drift"])
    return pd.DataFrame(rows)[["variable", "metric", "statistic", "p_value", "drift"]]

def psi_semaforo(psi_val: float) -> str:
    """Semaforo PSI: verde < 0.1, amarillo 0.1-0.25, rojo > 0.25."""
    if psi_val < 0.1:
        return "VERDE"
    elif psi_val <= 0.25:
        return "AMARILLO"
    return "ROJO"

def psi_color(psi_val: float) -> str:
    """Color CSS para el semaforo PSI."""
    if psi_val < 0.1:
        return "#28a745"
    elif psi_val <= 0.25:
        return "#ffc107"
    return "#dc3545"

# Reporte Evidently (opcional)

def generate_evidently_report(df_ref: pd.DataFrame, df_cur: pd.DataFrame):
    """Genera reporte HTML de data drift con Evidently."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        cols = [c for c in ALL_FEATURES if c in df_ref.columns and c in df_cur.columns]
        report = Report([DataDriftPreset()])
        report.run(reference_data=df_ref[cols], current_data=df_cur[cols])

        output_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "monitoring_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = output_dir / f"evidently_drift_{ts}.html"
        report.save_html(str(html_path))
        return str(html_path)
    except ImportError:
        st.warning("Evidently no esta instalado. Ejecuta: pip install evidently")
        return None

# Aplicacion Streamlit

def main():
    st.set_page_config(page_title="Model Monitoring", layout="wide")

    st.title("Model Monitoring - Data Drift")
    st.caption("Pipeline de Credito | Monitoreo y deteccion de cambios poblacionales")

    # -- Cargar datos --
    X_ref, X_new, y_ref, y_new = load_data()

    # -- Sidebar --
    st.sidebar.header("Configuracion")
    st.sidebar.markdown(f"**Referencia:** {len(X_ref):,} registros")
    st.sidebar.markdown(f"**Nuevos datos:** {len(X_new):,} registros")

    # Estado API
    st.sidebar.markdown("---")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.sidebar.success("API conectada")
        st.sidebar.json(health)
    except Exception:
        st.sidebar.error("API no disponible")
        st.sidebar.caption("Ejecuta: uvicorn main:app --port 8000")

    # Muestreo
    st.sidebar.markdown("---")
    st.sidebar.subheader("Muestreo periodico")
    sample_size = st.sidebar.slider(
        "Tamano de muestra", min_value=10,
        max_value=min(500, len(X_new)), value=min(100, len(X_new)), step=10,
    )

    if st.sidebar.button("Generar nuevas predicciones", type="primary", use_container_width=True):
        X_sample = X_new.sample(n=sample_size, random_state=int(time.time()) % 10000)
        with st.sidebar, st.spinner("Consultando API..."):
            result = get_predictions(X_sample)
        if result:
            log_predictions(X_sample, result["predictions"], result["probabilities"])
            st.sidebar.success(f"{len(result['predictions'])} predicciones registradas")
            time.sleep(0.3)
            st.rerun()
        else:
            st.sidebar.error("No se pudieron generar predicciones")

    # -- Cargar log --
    logged_data = load_log()

    # ==================================================================
    # TABS
    # ==================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "Metricas y Graficas",
        "Data Drift",
        "Analisis Temporal",
        "Recomendaciones",
    ])

    # ==================================================================
    # TAB 1: Visualizacion de metricas
    # ==================================================================
    with tab1:
        if logged_data.empty or "prediction" not in logged_data.columns:
            st.info("No hay predicciones registradas. Usa el boton 'Generar nuevas predicciones' en el sidebar.")
        else:
            probs = pd.to_numeric(logged_data["probability"], errors="coerce").dropna()
            preds = pd.to_numeric(logged_data["prediction"], errors="coerce").dropna()

            # Metricas principales
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total predicciones", f"{len(preds):,}")
            m2.metric("Promedio probabilidad", f"{probs.mean():.4f}")
            m3.metric("Desviacion estandar", f"{probs.std():.4f}")
            m4.metric("Tasa positiva (>0.5)", f"{(probs > 0.5).mean():.2%}")

            st.markdown("---")

            # Histograma de predicciones
            st.subheader("Distribucion de probabilidades predichas")
            fig_hist = px.histogram(
                logged_data, x="probability", nbins=30,
                title="Histograma de probabilidades",
                labels={"probability": "Probabilidad", "count": "Frecuencia"},
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

            # Comparacion de medias
            st.subheader("Comparacion de medias: Referencia vs Actual")
            feature_cols = [c for c in NUMERIC_FEATURES if c in logged_data.columns and c in X_ref.columns]
            comp_data = []
            for col in feature_cols:
                comp_data.append({
                    "Feature": col,
                    "Referencia": round(X_ref[col].mean(), 2),
                    "Actual": round(pd.to_numeric(logged_data[col], errors="coerce").mean(), 2),
                })

            if comp_data:
                comp_df = pd.DataFrame(comp_data)
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name="Referencia", x=comp_df["Feature"], y=comp_df["Referencia"],
                    marker_color="lightblue",
                ))
                fig_comp.add_trace(go.Bar(
                    name="Actual", x=comp_df["Feature"], y=comp_df["Actual"],
                    marker_color="orange",
                ))
                fig_comp.update_layout(
                    title="Comparacion de medias por variable",
                    barmode="group", height=500, xaxis_tickangle=-45,
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            # Log reciente
            st.subheader("Ultimas predicciones registradas")
            n_rows = st.selectbox("Mostrar ultimas filas:", [10, 25, 50, 100], index=1)
            st.dataframe(logged_data.tail(n_rows), use_container_width=True, hide_index=True)

            st.caption(f"Total registros en log: {len(logged_data):,}")
            csv_data = logged_data.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar CSV completo", csv_data, "monitoring_log.csv", "text/csv")

    # ==================================================================
    # TAB 2: Data Drift
    # ==================================================================
    with tab2:
        if logged_data.empty:
            st.info("No hay datos de monitoreo. Genera predicciones primero.")
        else:
            drift_current = logged_data.drop(columns=["prediction", "probability", "timestamp"], errors="ignore")
            drift_report = compute_drift_report(X_ref, drift_current)

            if drift_report.empty:
                st.warning("No se pudieron calcular metricas de drift.")
            else:
                # -- Semaforo PSI --
                st.subheader("Semaforo PSI por variable")
                psi_rows = drift_report[drift_report["metric"] == "PSI"]
                if not psi_rows.empty:
                    cols = st.columns(min(len(psi_rows), 4))
                    for i, (_, row) in enumerate(psi_rows.iterrows()):
                        color = psi_color(row["statistic"])
                        label = psi_semaforo(row["statistic"])
                        cols[i % len(cols)].markdown(
                            f'<div style="background:{color};color:white;padding:10px;border-radius:5px;'
                            f'text-align:center;margin-bottom:8px;">'
                            f'<b>{row["variable"]}</b><br>PSI = {row["statistic"]:.4f}<br>{label}</div>',
                            unsafe_allow_html=True,
                        )

                st.markdown("---")

                # -- Tabla de metricas con colores --
                st.subheader("Metricas de drift por variable")

                def style_drift(val):
                    if val is True:
                        return "background-color: #ffcccc; color: #cc0000; font-weight: bold"
                    elif val is False:
                        return "background-color: #ccffcc; color: #006600"
                    return ""

                styled = drift_report.style.map(style_drift, subset=["drift"])
                st.dataframe(styled, use_container_width=True, height=500)

                # -- Features con drift --
                st.markdown("---")
                vars_with_drift = drift_report[drift_report["drift"]]["variable"].nunique()
                total_vars = drift_report["variable"].nunique()
                d1, d2 = st.columns(2)
                d1.metric("Features con drift", f"{vars_with_drift} / {total_vars}")
                d2.metric("% Features con drift", f"{vars_with_drift / max(total_vars, 1) * 100:.1f}%")

                # -- Distribucion historica vs actual (numericas) --
                st.markdown("---")
                st.subheader("Distribucion historica vs actual")
                selected_var = st.selectbox("Selecciona variable numerica", NUMERIC_FEATURES, key="drift_num")

                if selected_var in X_ref.columns and selected_var in drift_current.columns:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=X_ref[selected_var].dropna(), name="Referencia",
                        opacity=0.6, marker_color="lightblue", histnorm="probability density",
                    ))
                    fig.add_trace(go.Histogram(
                        x=pd.to_numeric(drift_current[selected_var], errors="coerce").dropna(),
                        name="Actual", opacity=0.6, marker_color="orange", histnorm="probability density",
                    ))
                    fig.update_layout(barmode="overlay", title=f"Distribucion: {selected_var}", height=400)
                    st.plotly_chart(fig, use_container_width=True)

                # -- Distribucion categoricas --
                st.subheader("Distribucion categoricas: Referencia vs Actual")
                cat_var = st.selectbox("Selecciona variable categorica", CATEGORICAL_FEATURES + ORDINAL_FEATURES, key="drift_cat")

                if cat_var in X_ref.columns and cat_var in drift_current.columns:
                    ref_c = X_ref[cat_var].dropna().astype(str).value_counts(normalize=True).rename("Referencia")
                    cur_c = drift_current[cat_var].dropna().astype(str).value_counts(normalize=True).rename("Actual")
                    comp = pd.concat([ref_c, cur_c], axis=1).fillna(0)

                    fig_cat = go.Figure()
                    fig_cat.add_trace(go.Bar(x=comp.index, y=comp["Referencia"], name="Referencia", marker_color="lightblue"))
                    fig_cat.add_trace(go.Bar(x=comp.index, y=comp["Actual"], name="Actual", marker_color="orange"))
                    fig_cat.update_layout(barmode="group", title=f"Distribucion: {cat_var}", height=400)
                    st.plotly_chart(fig_cat, use_container_width=True)

                # -- Barras de riesgo --
                st.markdown("---")
                st.subheader("Indicador de riesgo por variable")
                risk_data = []
                for var in drift_report["variable"].unique():
                    var_df = drift_report[drift_report["variable"] == var]
                    score = var_df["drift"].sum() / len(var_df) * 100
                    risk_data.append({"variable": var, "riesgo": score})

                df_risk = pd.DataFrame(risk_data).sort_values("riesgo", ascending=True)
                fig_risk = px.bar(
                    df_risk, x="riesgo", y="variable", orientation="h",
                    color="riesgo", color_continuous_scale=["green", "yellow", "red"],
                    range_color=[0, 100], title="Score de riesgo (% metricas con drift)",
                    labels={"riesgo": "Riesgo (%)", "variable": "Variable"},
                )
                fig_risk.update_layout(height=500)
                st.plotly_chart(fig_risk, use_container_width=True)

    # ==================================================================
    # TAB 3: Analisis temporal
    # ==================================================================
    with tab3:
        if logged_data.empty or "timestamp" not in logged_data.columns:
            st.info("No hay datos temporales. Genera predicciones varias veces para ver la evolucion.")
        else:
            logged_data["timestamp_dt"] = pd.to_datetime(logged_data["timestamp"], errors="coerce")
            ts_valid = logged_data.dropna(subset=["timestamp_dt"])

            if ts_valid.empty:
                st.warning("No se encontraron timestamps validos.")
            else:
                # Evolucion de probabilidad promedio
                st.subheader("Evolucion de la probabilidad promedio en el tiempo")
                ts_grouped = ts_valid.groupby("timestamp_dt").agg(
                    prob_mean=("probability", "mean"),
                    prob_std=("probability", "std"),
                    count=("prediction", "count"),
                ).reset_index()

                fig_time = px.line(
                    ts_grouped, x="timestamp_dt", y="prob_mean",
                    title="Probabilidad promedio por muestreo",
                    labels={"timestamp_dt": "Fecha", "prob_mean": "Probabilidad promedio"},
                    markers=True,
                )
                fig_time.update_layout(height=400)
                st.plotly_chart(fig_time, use_container_width=True)

                # Deteccion de cambios abruptos
                st.subheader("Deteccion de cambios abruptos")
                if len(ts_grouped) >= 3:
                    ts_grouped["diff"] = ts_grouped["prob_mean"].diff().abs()
                    umbral = ts_grouped["diff"].mean() + 2 * ts_grouped["diff"].std()
                    abrupt = ts_grouped[ts_grouped["diff"] > umbral]

                    if not abrupt.empty:
                        st.warning(f"Se detectaron {len(abrupt)} cambios abruptos en las predicciones.")
                        st.dataframe(abrupt[["timestamp_dt", "prob_mean", "diff"]], use_container_width=True, hide_index=True)
                    else:
                        st.success("No se detectaron cambios abruptos.")
                else:
                    st.info("Se necesitan al menos 3 muestreos para detectar cambios abruptos.")

                # Evolucion del drift por variable
                st.subheader("Evolucion del drift a lo largo del tiempo")
                st.caption("Para ver la evolucion, genera predicciones en diferentes momentos.")

                # Calcular drift actual vs referencia particionando el log por timestamp
                timestamps = sorted(ts_valid["timestamp_dt"].unique())
                if len(timestamps) >= 2:
                    drift_evolution = []
                    for ts in timestamps:
                        batch = ts_valid[ts_valid["timestamp_dt"] == ts].drop(
                            columns=["prediction", "probability", "timestamp", "timestamp_dt"], errors="ignore"
                        )
                        if len(batch) < 5:
                            continue
                        for col in NUMERIC_FEATURES:
                            if col in batch.columns and col in X_ref.columns:
                                ks_result = ks_test(X_ref[col], pd.to_numeric(batch[col], errors="coerce"))
                                drift_evolution.append({
                                    "timestamp": ts, "variable": col,
                                    "KS_statistic": ks_result["statistic"],
                                })

                    if drift_evolution:
                        df_evo = pd.DataFrame(drift_evolution)
                        fig_evo = px.line(
                            df_evo, x="timestamp", y="KS_statistic", color="variable",
                            title="Evolucion del KS statistic por variable",
                            markers=True,
                        )
                        fig_evo.update_layout(height=450)
                        st.plotly_chart(fig_evo, use_container_width=True)

    # ==================================================================
    # TAB 4: Recomendaciones
    # ==================================================================
    with tab4:
        if logged_data.empty:
            st.info("No hay datos de monitoreo. Genera predicciones primero.")
        else:
            drift_current = logged_data.drop(columns=["prediction", "probability", "timestamp"], errors="ignore")
            drift_report = compute_drift_report(X_ref, drift_current)

            if drift_report.empty:
                st.info("No se pudieron calcular metricas de drift.")
            else:
                st.subheader("Recomendaciones automaticas")

                n_drift = drift_report["drift"].sum()
                n_total = len(drift_report)
                pct = n_drift / max(n_total, 1) * 100
                vars_affected = drift_report[drift_report["drift"]]["variable"].unique().tolist()

                if pct == 0:
                    st.success(
                        "No se detecto drift en ninguna variable. "
                        "El modelo opera dentro de los parametros esperados."
                    )
                elif pct < 20:
                    st.info(
                        f"Drift leve detectado en {len(vars_affected)} variable(s): "
                        f"{', '.join(vars_affected)}. "
                        "Se recomienda monitorear estas variables con mayor frecuencia."
                    )
                elif pct < 50:
                    st.warning(
                        f"Drift moderado ({pct:.0f}% de metricas). "
                        f"Variables afectadas: {', '.join(vars_affected)}. "
                        "Considerar re-evaluar el desempeno del modelo con datos recientes."
                    )
                else:
                    st.error(
                        f"Drift significativo ({pct:.0f}% de metricas). "
                        f"Variables afectadas: {', '.join(vars_affected)}. "
                        "SE RECOMIENDA REENTRENAR EL MODELO con datos actualizados."
                    )

                # PSI critico
                psi_critical = drift_report[
                    (drift_report["metric"] == "PSI") & (drift_report["statistic"] > 0.25)
                ]
                if not psi_critical.empty:
                    st.error(
                        f"PSI critico (>0.25) en: {', '.join(psi_critical['variable'].unique())}. "
                        "Cambio poblacional severo detectado. Revision inmediata requerida."
                    )

                # KS significativo
                ks_drift = drift_report[
                    (drift_report["metric"] == "KS") & (drift_report["drift"])
                ]
                if not ks_drift.empty:
                    st.warning(
                        f"Test KS significativo en: {', '.join(ks_drift['variable'].unique())}. "
                        "La distribucion de estas variables ha cambiado estadisticamente."
                    )

                # Sugerencias
                st.markdown("---")
                st.subheader("Sugerencias")
                st.markdown("""
                - **Si el drift es leve**: aumentar la frecuencia de monitoreo y observar tendencia.
                - **Si el drift es moderado**: re-evaluar metricas de desempeno del modelo (AUC, F1) con datos recientes.
                - **Si el drift es severo**: reentrenar el modelo con datos actualizados, revisando las variables criticas.
                - **Variables con PSI > 0.25**: investigar causa raiz del cambio poblacional (cambios regulatorios, estacionalidad, crisis economica).
                """)

                # Evidently
                st.markdown("---")
                if st.button("Generar Reporte Evidently (HTML)"):
                    with st.spinner("Generando reporte..."):
                        path = generate_evidently_report(X_ref, drift_current)
                    if path:
                        st.success(f"Reporte guardado en: {path}")

    # -- Limpiar log (al fondo del sidebar) --
    st.sidebar.markdown("---")
    if st.sidebar.button("Limpiar log de monitoreo"):
        Path(MONITOR_LOG).unlink(missing_ok=True)
        st.sidebar.success("Log eliminado.")
        st.rerun()

if __name__ == "__main__":
    main()
