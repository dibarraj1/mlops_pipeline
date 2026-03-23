# main.py

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Configuración

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL = "xgboost"

app = FastAPI(
    title="API Predicción de Crédito",
    description="Predicción de pago a tiempo — Pipeline MLOps",
    version="1.0.0",
)

# Estado global

model_artifact = None
preprocessor = None
threshold = 0.5
model_version = DEFAULT_MODEL

# Carga de artefactos

def load_artifacts(model_name: str = DEFAULT_MODEL):
    """Carga modelo, preprocessor y threshold desde disco."""
    global model_artifact, preprocessor, threshold, model_version

    model_path = MODELS_DIR / f"{model_name}.joblib"
    preprocessor_path = MODELS_DIR / "preprocessor.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor no encontrado: {preprocessor_path}")

    artifact = joblib.load(model_path)
    model_artifact = artifact["model"]
    threshold = artifact.get("threshold", 0.5)
    preprocessor = joblib.load(preprocessor_path)
    model_version = model_name

    logger.info(f"Modelo '{model_name}' cargado (threshold={threshold:.2f})")


@app.on_event("startup")
def startup():
    try:
        load_artifacts()
    except FileNotFoundError as e:
        logger.warning(f"No se pudo cargar modelo al iniciar: {e}")
        logger.warning("Ejecuta model_training.py primero para generar los artefactos.")

# Schemas Pydantic

class CreditRequest(BaseModel):
    puntaje_datacredito: float | None = None
    huella_consulta: float | None = None
    capital_prestado: float | None = None
    plazo_meses: float | None = None
    cuota_pactada: float | None = None
    salario_cliente: float | None = None
    total_otros_prestamos: float | None = None
    promedio_ingresos_datacredito: float | None = None
    edad_cliente: float | None = None
    cant_creditosvigentes: float | None = None
    creditos_sectorFinanciero: float | None = None
    creditos_sectorCooperativo: float | None = None
    creditos_sectorReal: float | None = None
    tipo_credito: str | int | None = None
    tipo_laboral: str | None = None
    tendencia_ingresos: str | None = None


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    threshold: float


class BatchResponse(BaseModel):
    predictions: list[int]
    probabilities: list[float]
    threshold: float

# Endpoints

@app.get("/health")
def health():
    """Health check del servicio."""
    return {
        "status": "ok",
        "model_loaded": model_artifact is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_version": model_version,
        "threshold": threshold,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: CreditRequest):
    """Predicción individual — un solo registro."""
    if model_artifact is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Ejecuta model_training.py primero.",
        )

    df = pd.DataFrame([request.model_dump()])
    df = df.fillna(np.nan)

    try:
        X = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en preprocesamiento: {e}")

    proba = float(model_artifact.predict_proba(X)[0, 1])
    pred = int(proba >= threshold)

    return PredictionResponse(
        prediction=pred,
        probability=round(proba, 4),
        threshold=threshold,
    )


@app.post("/predict_batch", response_model=BatchResponse)
def predict_batch(requests: list[CreditRequest]):
    """Predicción en lote — múltiples registros."""
    if model_artifact is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Ejecuta model_training.py primero.",
        )

    if not requests:
        raise HTTPException(status_code=400, detail="La lista de registros está vacía.")

    data = [r.model_dump() for r in requests]
    df = pd.DataFrame(data)
    df = df.fillna(np.nan)

    try:
        X = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en preprocesamiento: {e}")

    probas = model_artifact.predict_proba(X)[:, 1]
    preds = (probas >= threshold).astype(int)

    return BatchResponse(
        predictions=preds.tolist(),
        probabilities=[round(float(p), 4) for p in probas],
        threshold=threshold,
    )

# Ejecución directa

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)