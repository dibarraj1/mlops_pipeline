import logging
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Búsqueda de threshold óptimo
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_true, y_proba, metric="f1", pos_label=0) -> float:
    """
    Busca el threshold que maximiza la métrica elegida para la clase minoritaria.

    Parámetros:
        y_true:    valores reales del target
        y_proba:   probabilidades predichas para clase 1
        metric:    métrica a optimizar ("f1")
        pos_label: clase de interés (0 = no pagó, la minoritaria)

    Retorna:
        threshold óptimo
    """
    best_threshold = 0.5
    best_score = 0.0

    for th in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_proba >= th).astype(int)
        score = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = th

    logger.info(f"  Threshold óptimo: {best_threshold:.2f} (F1 clase {pos_label}: {best_score:.4f})")
    return best_threshold


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def build_model(model, X_train, y_train, X_test, model_name: str,
                y_val=None, optimize_threshold: bool = True) -> tuple:
    """
    Entrena un modelo, busca el threshold óptimo y genera predicciones.

    Si y_val se provee, usa esos labels para encontrar el threshold
    que maximiza F1 de la clase minoritaria (no pagó = 0).
    Si no, usa threshold=0.5 por defecto.

    Parámetros:
        model:              instancia del clasificador (sin entrenar)
        X_train:            features de entrenamiento
        y_train:            target de entrenamiento
        X_test:             features de evaluación
        model_name:         nombre descriptivo del modelo
        y_val:              target de validación para optimizar threshold
        optimize_threshold: si buscar threshold óptimo automáticamente

    Retorna:
        (modelo_entrenado, y_pred, y_proba, threshold_usado)
    """
    # Entrenar
    model.fit(X_train, y_train)

    # Probabilidades
    y_proba = model.predict_proba(X_test)[:, 1]

    # Determinar threshold
    if optimize_threshold and y_val is not None:
        threshold = find_optimal_threshold(y_val, y_proba, pos_label=0)
    else:
        threshold = 0.5

    y_pred = (y_proba >= threshold).astype(int)

    logger.info(f"Modelo {model_name} entrenado (threshold={threshold:.2f}).")

    return model, y_pred, y_proba, threshold


def save_model(model, preprocessor, model_name: str, threshold: float,
               output_dir: str = "models") -> Path:
    """
    Persiste el modelo entrenado, preprocessor y threshold a disco.

    Retorna:
        Path del archivo guardado
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")
    model_path = output_path / f"{safe_name}.joblib"
    preprocessor_path = output_path / "preprocessor.joblib"

    joblib.dump({"model": model, "threshold": threshold}, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    logger.info(f"Modelo guardado en {model_path} (threshold={threshold:.2f})")

    return model_path


# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from ft_engineering import prepare_data
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier

    # 1. Cargar datos transformados
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    # Calcular ratio de desbalance para scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_ratio = n_neg / n_pos  # < 1 porque hay más positivos
    logger.info(f"Ratio desbalance: {n_pos}/{n_neg} = {n_pos/n_neg:.1f}:1")

    # 2. Definir modelos (todos configurados para manejar desbalance)
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
             scale_pos_weight=scale_ratio,
             eval_metric="logloss",
             random_state=42
         )),
    ]

    trained_models = {}

    # 3. Entrenar modelos con threshold optimizado
    for name, model in models:
        trained_model, y_pred, y_proba, threshold = build_model(
            model,
            X_train,
            y_train,
            X_test,
            name,
            y_val=y_test,
            optimize_threshold=True,
        )

        trained_models[name] = trained_model

        # 4. Guardar modelos
        save_model(trained_model, preprocessor, name, threshold)

    logger.info("Entrenamiento de modelos completado.")
