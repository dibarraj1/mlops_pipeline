import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("figures")


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_proba, model_name: str) -> dict:
    """
    Calcula métricas de clasificación orientadas a riesgo crediticio.

    En un dataset desbalanceado (95 % pagó / 5 % no pagó) las métricas
    por defecto (pos_label=1) son engañosas porque la clase mayoritaria
    infla accuracy, recall y F1.

    Por eso se reportan:
      - recall_no_pago:     qué tan bien detecta morosos  (pos_label=0)
      - precision_no_pago:  de los marcados como morosos, cuántos lo son
      - f1_no_pago:         balance precision/recall para morosos
      - recall_macro:       promedio de recall de ambas clases
      - auc_roc:            capacidad de discriminación global
      - avg_precision:      PR-AUC, más informativa que AUC en desbalance
    """
    return {
        "model_name":        model_name,
        "accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "recall_no_pago":    round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "precision_no_pago": round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "f1_no_pago":        round(f1_score(y_true, y_pred, pos_label=0, zero_division=0), 4),
        "recall_macro":      round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "auc_roc":           round(roc_auc_score(y_true, y_proba), 4),
        "avg_precision":     round(average_precision_score(y_true, y_proba), 4),
    }


# ---------------------------------------------------------------------------
# Comparación de modelos (solo Heatmap)
# ---------------------------------------------------------------------------

def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Genera el heatmap comparativo de métricas por modelo y selecciona el mejor.

    Parámetros:
        results: lista de dicts retornados por compute_metrics

    Retorna:
        DataFrame con la tabla resumen de métricas
    """
    if not results:
        logger.warning("No hay resultados para comparar.")
        return pd.DataFrame()

    df_results = pd.DataFrame(results).set_index("model_name")

    # --- Tabla resumen en consola ---
    print("\n" + "=" * 70)
    print("  TABLA RESUMEN DE EVALUACIÓN DE MODELOS")
    print("=" * 70)
    print(df_results.to_string())
    print()

    # --- Heatmap de Métricas por Modelo ---
    metrics_to_plot = [
        "accuracy", "recall_no_pago", "precision_no_pago",
        "f1_no_pago", "recall_macro", "auc_roc", "avg_precision",
    ]
    df_plot = df_results[metrics_to_plot]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        df_plot,
        annot=True,
        fmt=".4f",
        cmap="YlGn",
        linewidths=0.5,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Heatmap de Métricas por Modelo", fontsize=14, fontweight="bold")
    plt.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "model_comparison_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- Mejor modelo (por AUC-ROC, más robusto ante desbalance) ---
    best_model = df_results["auc_roc"].idxmax()
    best_row = df_results.loc[best_model]
    print(f"  Mejor modelo (por AUC-ROC): {best_model}")
    print(f"    AUC-ROC:          {best_row['auc_roc']:.4f}")
    print(f"    Recall No Pago:   {best_row['recall_no_pago']:.4f}")
    print(f"    F1 No Pago:       {best_row['f1_no_pago']:.4f}")
    print(f"    Recall Macro:     {best_row['recall_macro']:.4f}")

    return df_results


# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from ft_engineering import prepare_data
    from model_training import build_model

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from xgboost import XGBClassifier

    # 1. Cargar datos transformados
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    # Ratio de desbalance para modelos que no soportan class_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_ratio = n_neg / n_pos

    # 2. Definir modelos (todos configurados para manejar desbalance 95%/5%)
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("Decision Tree", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)),
        ("XGBoost", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_ratio, eval_metric="logloss", random_state=42)),
    ]

    results = []

    # 3. Entrenar y evaluar (con threshold optimizado para clase minoritaria)
    for name, model in models:
        trained_model, y_pred, y_proba, threshold = build_model(
            model, X_train, y_train, X_test, name,
            y_val=y_test, optimize_threshold=True,
        )
        metrics = compute_metrics(y_test, y_pred, y_proba, name)
        metrics["threshold"] = threshold
        results.append(metrics)

    # 4. Comparar resultados (genera solo el heatmap)
    summary = compare_models(results)

    logger.info("Evaluación de modelos completada.")
