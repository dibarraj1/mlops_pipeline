

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


NUMERIC_FEATURES = [
    "capital_prestado", "salario_cliente", "total_otros_prestamos",
    "cuota_pactada", "puntaje_datacredito",
    "saldo_mora", "saldo_total", "saldo_principal", "saldo_mora_codeudor",
    "promedio_ingresos_datacredito",
    "plazo_meses", "edad_cliente", "cant_creditosvigentes", "huella_consulta",
    "creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal",
    "ratio_cuota_salario", "ratio_deuda_salario", "ratio_capital_plazo",
]

CATEGORICAL_FEATURES = ["tipo_credito", "tipo_laboral"]

ORDINAL_FEATURES = ["tendencia_ingresos"]

ORDINAL_CATEGORIES = [["Decreciente", "Estable", "Creciente"]]

TARGET = "Pago_atiempo"

VALID_TENDENCIA = {"Creciente", "Decreciente", "Estable"}



# Funciones de carga y limpieza

def load_and_clean_data(filepath: str = "Base_de_datos.csv") -> pd.DataFrame:
    """Carga el CSV y realiza limpieza inicial de datos."""
    df = pd.read_csv(filepath)

    # Limpiar tendencia_ingresos: solo valores válidos, el resto NaN
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(VALID_TENDENCIA)
    )

    # Eliminar fecha_prestamo (no se usa en modelamiento)
    df = df.drop(columns=["fecha_prestamo"])

    # Convertir tipo_credito a string para que el encoder lo trate como categórico
    df["tipo_credito"] = df["tipo_credito"].astype(str)

    return df



# Creación de features derivados

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Genera variables derivadas a partir de las existentes."""
    df = df.copy()

    # Ratio cuota / salario (carga financiera mensual)
    df["ratio_cuota_salario"] = np.where(
        df["salario_cliente"] > 0,
        df["cuota_pactada"] / df["salario_cliente"],
        np.nan,
    )

    # Ratio deuda total / salario (nivel de endeudamiento)
    df["ratio_deuda_salario"] = np.where(
        df["salario_cliente"] > 0,
        df["total_otros_prestamos"] / df["salario_cliente"],
        np.nan,
    )

    # Ratio capital / plazo (intensidad del préstamo por mes)
    df["ratio_capital_plazo"] = df["capital_prestado"] / df["plazo_meses"]

    return df


# Construcción del ColumnTransformer (pipeline de preprocesamiento), remplaza nulls y demas valores de las columnas numericas y categoricas respectivamente, ademas de codificar las categoricas con onehotencoder y ordinalencoder segun corresponda

def build_preprocessor() -> ColumnTransformer:
    """
    Construye el ColumnTransformer con tres ramas:
      - numeric:            SimpleImputer(median)
      - categoric:          SimpleImputer(most_frequent) → OneHotEncoder
      - categoric_ordinal:  SimpleImputer(most_frequent) → OrdinalEncoder
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=ORDINAL_CATEGORIES)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categoric", categorical_pipeline, CATEGORICAL_FEATURES),
            ("categoric_ordinal", ordinal_pipeline, ORDINAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor



# Función principal: preparar datos para modelamiento

def prepare_data(
    filepath: str = "Base_de_datos.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Pipeline completo de feature engineering.

    Retorna:
        X_train, X_test, y_train, y_test, preprocessor
    """
    # 1. Cargar y limpiar
    df = load_and_clean_data(filepath)

    # 2. Crear features derivados
    df = create_features(df)

    # 3. Separar X e y
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 4. Train/test split estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 5. Construir y ajustar el preprocessor
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    print(f"Datos de entrenamiento: {X_train_transformed.shape}")
    print(f"Datos de evaluación:    {X_test_transformed.shape}")
    print(f"Distribución target train: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"Distribución target test:  {y_test.value_counts(normalize=True).to_dict()}")

    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor



# Ejecución directa (para pruebas)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    print("\nFeature engineering completado exitosamente.")
    print(f"Features generados: {X_train.shape[1]}")
