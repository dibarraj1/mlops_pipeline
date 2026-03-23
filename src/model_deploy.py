# model_deploy.py
# API de prediccion con FastAPI
# Ejecutar con: uvicorn model_deploy:app --reload --port 8000

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Configuracion de rutas
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = BASE_DIR.parent / "models"

app = FastAPI(
    title="API Prediccion de Credito",
    description="Prediccion de pago a tiempo de creditos",
    version="1.0.0",
)

# Carga de artefactos
model = None
preprocessor = None
threshold = 0.5

def load_artifacts():
    global model, preprocessor, threshold
    model_path = MODELS_DIR / "xgboost.joblib"
    prep_path = MODELS_DIR / "preprocessor.joblib"

    if not model_path.exists() or not prep_path.exists():
        return False

    bundle = joblib.load(model_path)
    model = bundle["model"]
    threshold = bundle.get("threshold", 0.5)
    preprocessor = joblib.load(prep_path)
    return True

@app.on_event("startup")
def startup():
    success = load_artifacts()
    if not success:
        print("ADVERTENCIA: No se encontraron los modelos en", MODELS_DIR)
        print("Ejecuta model_training.py primero.")

# Schema
class CreditRequest(BaseModel):
    puntaje_datacredito: Optional[float] = None
    huella_consulta: Optional[float] = None
    capital_prestado: Optional[float] = None
    plazo_meses: Optional[float] = None
    cuota_pactada: Optional[float] = None
    salario_cliente: Optional[float] = None
    total_otros_prestamos: Optional[float] = None
    promedio_ingresos_datacredito: Optional[float] = None
    edad_cliente: Optional[float] = None
    cant_creditosvigentes: Optional[float] = None
    creditos_sectorFinanciero: Optional[float] = None
    creditos_sectorCooperativo: Optional[float] = None
    creditos_sectorReal: Optional[float] = None
    tipo_credito: Optional[str] = None
    tipo_laboral: Optional[str] = None
    tendencia_ingresos: Optional[str] = None

# Endpoints
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_version": "xgboost",
        "threshold": threshold,
    }

@app.post("/predict")
def predict(data: CreditRequest):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    try:
        df = pd.DataFrame([data.model_dump()])
        X = preprocessor.transform(df)
        proba = float(model.predict_proba(X)[0, 1])
        pred = int(proba >= threshold)
        return {
            "prediction": pred,
            "probability": round(proba, 4),
            "threshold": threshold,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
def predict_batch(data: list[CreditRequest]):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    try:
        df = pd.DataFrame([row.model_dump() for row in data])
        X = preprocessor.transform(df)
        probas = model.predict_proba(X)[:, 1]
        preds = (probas >= threshold).astype(int)
        return {
            "predictions": preds.tolist(),
            "probabilities": [round(float(p), 4) for p in probas],
            "threshold": threshold,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
