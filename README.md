# Pipeline MLOps — Modelo de Riesgo Crediticio

## Caso de Negocio

Este proyecto implementa un pipeline completo de MLOps para un modelo de riesgo crediticio que predice si un cliente realizará el **pago a tiempo** de su crédito (`Pago_atiempo`).

El objetivo del negocio es reducir la tasa de morosidad identificando de forma temprana a los clientes con mayor riesgo de incumplimiento, permitiendo a la entidad financiera tomar decisiones informadas sobre aprobación de créditos y gestión de riesgo.

## Arquitectura del Sistema

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Base_de_datos   │────▶│  ft_engineering   │────▶│   model_training     │
│     .csv         │     │     .py           │     │       .py            │
└──────────────────┘     └──────────────────┘     └─────────┬────────────┘
                                                            │
                                                   models/*.joblib
                                                   preprocessor.joblib
                                                            │
                              ┌──────────────────────────────┤
                              │                              │
                    ┌─────────▼────────┐          ┌─────────▼────────────┐
                    │    main.py        │          │  model_monitoring.py  │
                    │   (FastAPI)       │◀─────────│    (Streamlit)        │
                    │   :8000           │  HTTP    │    :8501              │
                    └──────────────────┘          └───────────┬──────────┘
                                                              │
                                                   monitoring_log.csv
```

### Componentes

| Componente | Archivo | Descripción |
|---|---|---|
| Feature Engineering | `ft_engineering.py` | Limpieza, transformación y preprocesamiento |
| Entrenamiento | `model_training.py` | Entrenamiento de modelos (LR, DT, RF, GB, XGBoost) |
| Evaluación | `model_evaluation.py` | Métricas comparativas y selección del mejor modelo |
| API de Predicción | `main.py` | API REST con FastAPI para serving |
| Monitoreo | `model_monitoring.py` | Dashboard Streamlit con data drift y monitoreo |

## Features del Modelo

### Numéricas
| Feature | Descripción |
|---|---|
| `puntaje_datacredito` | Score crediticio del cliente |
| `huella_consulta` | Número de consultas en centrales de riesgo |
| `capital_prestado` | Monto del crédito otorgado |
| `plazo_meses` | Duración del crédito en meses |
| `cuota_pactada` | Valor de la cuota mensual |
| `salario_cliente` | Ingreso mensual del cliente |
| `total_otros_prestamos` | Deuda total en otros créditos |
| `promedio_ingresos_datacredito` | Promedio de ingresos según Datacrédito |
| `edad_cliente` | Edad del solicitante |
| `cant_creditosvigentes` | Número de créditos activos |
| `creditos_sectorFinanciero` | Créditos en sector financiero |
| `creditos_sectorCooperativo` | Créditos en sector cooperativo |
| `creditos_sectorReal` | Créditos en sector real |

### Categóricas
| Feature | Tipo | Valores |
|---|---|---|
| `tipo_credito` | Nominal | CONSUMO, COMERCIAL, MICROCREDITO, VIVIENDA |
| `tipo_laboral` | Nominal | ASALARIADO, INDEPENDIENTE, PENSIONADO, OTRO |
| `tendencia_ingresos` | Ordinal | Decreciente < Estable < Creciente |

### Variables excluidas (Data Leakage)
- `puntaje`, `saldo_mora`, `saldo_total`, `saldo_principal`, `saldo_mora_codeudor`

## API de Predicción (main.py)

### Endpoints

#### `GET /health`
Health check del servicio.

```json
{
  "status": "ok",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "model_version": "xgboost",
  "threshold": 0.42
}
```

#### `POST /predict`
Predicción individual. Recibe un JSON con las features y devuelve:

```json
{
  "prediction": 1,
  "probability": 0.7823,
  "threshold": 0.42
}
```

#### `POST /predict_batch`
Predicción en lote. Recibe una lista de registros y devuelve:

```json
{
  "predictions": [1, 0, 1, 1],
  "probabilities": [0.78, 0.31, 0.85, 0.62],
  "threshold": 0.42
}
```

## Dashboard de Monitoreo (model_monitoring.py)

### Tab 1: Online
- Formulario para predicción individual
- Envía datos a `POST /predict`
- Muestra prediction, probability y threshold

### Tab 2: Batch
- Upload de CSV para predicción masiva
- Validación de columnas
- Métricas agregadas
- Descarga de resultados

### Tab 3: Monitoring
Implementa el monitoreo completo del modelo en producción:

#### Métricas de Data Drift

| Métrica | Tipo | Umbral | Descripción |
|---|---|---|---|
| **Kolmogorov-Smirnov (KS)** | Numéricas | p < 0.05 | Compara las distribuciones acumuladas. Detecta cualquier diferencia en forma, ubicación o dispersión |
| **Population Stability Index (PSI)** | Numéricas | > 0.2 | Mide el cambio poblacional entre dos distribuciones. Estándar de la industria financiera |
| **Jensen-Shannon Divergence** | Numéricas | > 0.1 | Versión simétrica de KL divergence. Mide la similaridad entre distribuciones |
| **Chi-Cuadrado** | Categóricas | p < 0.05 | Compara frecuencias observadas vs esperadas en categorías |

#### Semáforo PSI
| Color | PSI | Interpretación |
|---|---|---|
| 🟢 Verde | < 0.1 | Sin cambio significativo |
| 🟡 Amarillo | 0.1 – 0.25 | Cambio moderado, monitorear |
| 🔴 Rojo | > 0.25 | Cambio significativo, considerar retraining |

#### Visualizaciones
- Histogramas comparativos (referencia vs actual)
- Comparación de medias por variable
- Barras por categoría
- Evolución temporal de predicciones
- Detección de cambios abruptos

#### Recomendaciones Automáticas
- Si 0% de métricas con drift: modelo estable
- Si < 20%: monitorear con mayor frecuencia
- Si < 50%: re-evaluar el modelo
- Si >= 50%: **reentrenar el modelo**
- Si PSI > 0.25: alerta crítica de cambio poblacional

## Instrucciones de Ejecución

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Entrenar modelos (si no existen los .joblib)

```bash
cd src
python model_training.py
```

Esto genera:
- `src/models/xgboost.joblib`
- `src/models/preprocessor.joblib`
- (y otros modelos)

### 3. Levantar la API

```bash
cd src
uvicorn main:app --reload --port 8000
```

Verificar en: http://localhost:8000/health

### 4. Levantar el Dashboard

En otra terminal:

```bash
cd src
streamlit run model_monitoring.py
```

Se abrirá en: http://localhost:8501

### Flujo de uso

1. Abrir el dashboard en el navegador
2. **Tab Online**: probar predicciones individuales
3. **Tab Batch**: subir un CSV y generar predicciones masivas
4. **Tab Monitoring**:
   - Usar el slider del sidebar para definir tamaño de muestra
   - Hacer clic en "Generar nuevas predicciones" varias veces
   - Observar las métricas de drift, gráficas y recomendaciones
   - Revisar los logs de predicciones

## Conclusiones y Recomendaciones

1. **Monitoreo continuo**: El data drift es inevitable en producción. Este sistema permite detectarlo antes de que afecte las decisiones de negocio.

2. **Umbral de retraining**: Cuando el PSI supera 0.25 en variables clave como `puntaje_datacredito` o `salario_cliente`, se recomienda reentrenar el modelo.

3. **Variables críticas**: Las variables financieras (capital, salario, puntaje) son las más susceptibles a drift por cambios macroeconómicos.

4. **Frecuencia de monitoreo**: Se recomienda ejecutar el monitoreo semanalmente en producción, o ante eventos externos relevantes (cambios regulatorios, crisis económicas).

5. **Logging**: El archivo `monitoring_log.csv` mantiene un registro histórico de todas las predicciones, permitiendo auditoría y trazabilidad.
