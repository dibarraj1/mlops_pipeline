


import numpy as np
import pandas as pd
from ft_engineering import prepare_data

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


# Función: build_model

def build_model(model, X_train, y_train, X_test, y_test, model_name: str, threshold: float = 0.4) -> tuple:
    """
    Entrena un modelo, genera predicciones y retorna el modelo con sus predicciones.

    Parámetros:
        model:      instancia del clasificador (sin entrenar)
        X_train:    features de entrenamiento
        y_train:    target de entrenamiento
        X_test:     features de evaluación
        y_test:     target de evaluación
        model_name: nombre descriptivo del modelo
        threshold:  umbral de clasificación (default 0.4)

    Retorna:
        (modelo_entrenado, y_pred, y_proba)
    """
    # Entrenar
    model.fit(X_train, y_train)

    # Predecir
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print(f"  Modelo {model_name} entrenado correctamente.")

    return model, y_pred, y_proba


# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Cargar datos transformados
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    # 2. Definir modelos
    models = [

    ("Logistic Regression",
     LogisticRegression(
         max_iter=1000,
         class_weight="balanced"
     )),

    ("Decision Tree",
     DecisionTreeClassifier(
         random_state=42,
         class_weight="balanced"
     )),

    ("Random Forest",
     RandomForestClassifier(
         n_estimators=200,
         random_state=42,
         class_weight="balanced"
     )),

    ("Gradient Boosting",
     GradientBoostingClassifier(
         n_estimators=200,
         learning_rate=0.05,
         max_depth=3,
         random_state=42
     )),

    ("XGBoost",
     XGBClassifier(
         n_estimators=300,
         learning_rate=0.05,
         max_depth=4,
         subsample=0.8,
         colsample_bytree=0.8,
         eval_metric="logloss",
         random_state=42
     )),
]

    trained_models = {}

    # 3. Entrenar modelos
    for name, model in models:
        trained_model, y_pred, y_proba = build_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            name
        )

        trained_models[name] = trained_model

    print("\nEntrenamiento de modelos completado.")
