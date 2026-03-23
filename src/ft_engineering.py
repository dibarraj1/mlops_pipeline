import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

logger = logging.getLogger(__name__)


# Variable objeivo
TARGET = "Pago_atiempo"

# Variables numéricas:
NUMERIC_FEATURES = [
    "puntaje_datacredito",
    "huella_consulta",
    "capital_prestado", "plazo_meses", "cuota_pactada",
    "salario_cliente", "total_otros_prestamos",
    "promedio_ingresos_datacredito",
    "edad_cliente", "cant_creditosvigentes",
    "creditos_sectorFinanciero", "creditos_sectorCooperativo", "creditos_sectorReal",
]
# Variables categóricas nominales
CATEGORICAL_FEATURES = ["tipo_credito", "tipo_laboral"]

# Variables categóricas ordinales
ORDINAL_FEATURES = ["tendencia_ingresos"]

# leaking features 
LEAKAGE_COLUMNS = [
    "puntaje", "saldo_mora", "saldo_total", "saldo_principal", "saldo_mora_codeudor",
]
ORDINAL_CATEGORIES = [["Decreciente", "Estable", "Creciente"]]
VALID_TENDENCIA = {"Creciente", "Decreciente", "Estable"}

def load_and_clean_data(filepath: str = "Base_de_datos.csv") -> pd.DataFrame:
    """Carga el CSV y realiza limpieza inicial de datos."""
    df = pd.read_csv(filepath)

    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(VALID_TENDENCIA)
    )

    df = df.drop(columns=["fecha_prestamo"], errors="ignore")

    cols_to_drop = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Columnas eliminadas por data leakage: {cols_to_drop}")

    df["tipo_credito"] = df["tipo_credito"].astype(str)

    return df


def build_preprocessor() -> ColumnTransformer:
    """
    Construye el ColumnTransformer con tres ramas:
      - numeric: SimpleImputer(median)
      - categoric: SimpleImputer(most_frequent) → OneHotEncoder
      - categoric_ordinal: SimpleImputer(most_frequent) → OrdinalEncoder
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


def prepare_data(
    filepath: str = "Base_de_datos.csv",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Pipeline completo de preparación de datos.

    Retorna:
        X_train, X_test, y_train, y_test, preprocessor
    """
    df = load_and_clean_data(filepath)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    logger.info(f"Datos de entrenamiento: {X_train_transformed.shape}")
    logger.info(f"Datos de evaluación:    {X_test_transformed.shape}")

    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    logger.info(f"Distribución target train: {train_dist.to_dict()}")
    logger.info(f"Distribución target test:  {test_dist.to_dict()}")
    return X_train_transformed, X_test_transformed, y_train, y_test, preprocessor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    logger.info("Preparación de datos completada exitosamente.")
    logger.info(f"Features generados: {X_train.shape[1]}")