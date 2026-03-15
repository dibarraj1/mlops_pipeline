


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)


# Función: summarize_classification

def summarize_classification(y_true, y_pred, y_proba, model_name: str) -> dict:
    """
    Genera un resumen completo de métricas de clasificación.

    Parámetros:
        y_true:     valores reales
        y_pred:     predicciones del modelo
        y_proba:    probabilidades de la clase positiva
        model_name: nombre del modelo para visualización

    Retorna:
        dict con métricas: accuracy, precision, recall, f1, auc_roc
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    # Imprimir reporte
    print(f"  Modelo: {model_name}")
    print(classification_report(y_true, y_pred, target_names=["No pagó (0)", "Pagó (1)"]))
    print(f"  AUC-ROC: {auc:.4f}")


    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap de la matriz de confusión
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
        xticklabels=["No pagó (0)", "Pagó (1)"],
        yticklabels=["No pagó (0)", "Pagó (1)"],
    )
    axes[0].set_title(f"Matriz de Confusión - {model_name}", fontweight="bold")
    axes[0].set_ylabel("Real")
    axes[0].set_xlabel("Predicción")

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.4f}")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"Curva ROC - {model_name}", fontweight="bold")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc, 4),
    }


# Función: compare_models

def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Genera gráficos comparativos y tabla resumen de todos los modelos.

    Parámetros:
        results: lista de dicts retornados por summarize_classification

    Retorna:
        DataFrame con la tabla resumen de métricas
    """
    df_results = pd.DataFrame(results)
    df_results = df_results.set_index("model_name")

    # --- Tabla resumen ---

    print("  TABLA RESUMEN DE EVALUACIÓN DE MODELOS")

    print(df_results.to_string())

    # --- Gráfico comparativo de barras ---
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    df_plot = df_results[metrics_to_plot]

    ax = df_plot.plot(
        kind="bar",
        figsize=(14, 6),
        edgecolor="white",
        width=0.8,
        colormap="Set2",
    )
    ax.set_title("Comparación de Métricas por Modelo", fontsize=14, fontweight="bold")
    ax.set_ylabel("Valor de la Métrica")
    ax.set_xlabel("Modelo")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Métricas", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.tick_params(axis="x", rotation=30)

    # Anotar valores sobre las barras
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, rotation=90, padding=3)

    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(figsize=(10, 5))
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
    plt.show()


    best_model = df_results["f1_score"].idxmax()
    best_f1 = df_results.loc[best_model, "f1_score"]
    best_auc = df_results.loc[best_model, "auc_roc"]
    print(f"Mejor modelo (por F1-Score): {best_model}")
    print(f"  F1-Score: {best_f1:.4f} | AUC-ROC: {best_auc:.4f}")

    return df_results


# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ft_engineering import prepare_data
    from model_training import build_model

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from xgboost import XGBClassifier

    # 1. Cargar datos transformados
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    # 2. Definir modelos
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ("Decision Tree", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)),
        ("XGBoost", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)),
    ]

    results = []

    # 3. Entrenar y evaluar modelos
    for name, model in models:
        trained_model, y_pred, y_proba = build_model(
            model, X_train, y_train, X_test, y_test, name
        )
        metrics = summarize_classification(y_test, y_pred, y_proba, name)
        results.append(metrics)

    # 4. Comparar resultados
    summary = compare_models(results)

    print("\nEvaluación de modelos completada.")
