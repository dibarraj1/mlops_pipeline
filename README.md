# Pipeline MLOps - Modelo de Riesgo Crediticio

## Caso de Negocio

Este proyecto implementa un pipeline completo de MLOps para un modelo de riesgo crediticio que predice si un cliente realizara el pago a tiempo de su credito (variable objetivo: `Pago_atiempo`).

El objetivo del negocio es reducir la tasa de morosidad identificando de forma temprana a los clientes con mayor riesgo de incumplimiento, permitiendo a la entidad financiera tomar decisiones informadas sobre aprobacion de creditos y gestion de riesgo.

## Arquitectura del Sistema

```
Base_de_datos.csv
       |
       v
ft_engineering.py  -->  model_training.py
                              |
                     models/xgboost.joblib
                     models/preprocessor.joblib
                              |
              +---------------+---------------+
              |                               |
       model_deploy.py                model_monitoring.py
        (FastAPI)                       (Streamlit)
        puerto 8000  <--- HTTP ---      puerto 8501
              |                               |
              v                               v
     Predicciones REST              monitoring_log.csv
```

### Componentes

| Componente | Archivo | Descripcion |
|---|---|---|
| Feature Engineering | `ft_engineering.py` | Limpieza, transformacion y preprocesamiento |
| Entrenamiento | `model_training.py` | Entrenamiento de modelos (LR, DT, RF, GB, XGBoost) |
| Evaluacion | `model_evaluation.py` | Metricas comparativas y seleccion del mejor modelo |
| API de Prediccion | `model_deploy.py` | API REST con FastAPI para serving del modelo |
| Monitoreo | `model_monitoring.py` | Dashboard Streamlit con data drift y monitoreo |

## Features del Modelo

### Numericas

| Feature | Descripcion |
|---|---|
| `puntaje_datacredito` | Score crediticio del cliente |
| `huella_consulta` | Numero de consultas en centrales de riesgo |
| `capital_prestado` | Monto del credito otorgado |
| `plazo_meses` | Duracion del credito en meses |
| `cuota_pactada` | Valor de la cuota mensual |
| `salario_cliente` | Ingreso mensual del cliente |
| `total_otros_prestamos` | Deuda total en otros creditos |
| `promedio_ingresos_datacredito` | Promedio de ingresos segun Datacredito |
| `edad_cliente` | Edad del solicitante |
| `cant_creditosvigentes` | Numero de creditos activos |
| `creditos_sectorFinanciero` | Creditos en sector financiero |
| `creditos_sectorCooperativo` | Creditos en sector cooperativo |
| `creditos_sectorReal` | Creditos en sector real |

### Categoricas

| Feature | Tipo | Valores |
|---|---|---|
| `tipo_credito` | Nominal | CONSUMO, COMERCIAL, MICROCREDITO, VIVIENDA |
| `tipo_laboral` | Nominal | ASALARIADO, INDEPENDIENTE, PENSIONADO, OTRO |
| `tendencia_ingresos` | Ordinal | Decreciente < Estable < Creciente |

### Variables excluidas por Data Leakage

- `puntaje`, `saldo_mora`, `saldo_total`, `saldo_principal`, `saldo_mora_codeudor`

Estas variables se excluyen porque contienen informacion que solo esta disponible despues de otorgado el credito, lo que generaria fuga de datos si se usan para predecir.

## API de Prediccion (model_deploy.py)

### Endpoints

#### GET /health

Health check del servicio.

```json
{
  "status": "ok",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "model_version": "xgboost",
  "threshold": 0.29
}
```

#### POST /predict

Prediccion individual. Recibe un JSON con las features y devuelve:

```json
{
  "prediction": 1,
  "probability": 0.7823,
  "threshold": 0.29
}
```

#### POST /predict_batch

Prediccion en lote. Recibe una lista de registros y devuelve:

```json
{
  "predictions": [1, 0, 1, 1],
  "probabilities": [0.78, 0.31, 0.85, 0.62],
  "threshold": 0.29
}
```

## Dashboard de Monitoreo (model_monitoring.py)

### Tab 1: Metricas y Graficas

- Metricas principales: total predicciones, promedio de probabilidad, desviacion estandar, tasa positiva
- Histograma de probabilidades predichas
- Comparacion de medias por variable (referencia vs actual)
- Log de predicciones recientes con opcion de descarga

### Tab 2: Data Drift

Implementa la deteccion de cambios poblacionales que puedan afectar el desempeno del modelo.

#### Metricas de Data Drift

| Metrica | Tipo de variable | Umbral de drift | Descripcion |
|---|---|---|---|
| Kolmogorov-Smirnov (KS) | Numericas | p-value < 0.05 | Compara las distribuciones acumuladas entre referencia y actual. Detecta diferencias en forma, ubicacion o dispersion de los datos |
| Population Stability Index (PSI) | Numericas | PSI > 0.2 | Mide el cambio poblacional entre dos distribuciones. Es el estandar de la industria financiera para monitoreo de modelos |
| Jensen-Shannon Divergence | Numericas | JS > 0.1 | Version simetrica de la divergencia de Kullback-Leibler. Cuantifica la similaridad entre dos distribuciones de probabilidad |
| Chi-Cuadrado | Categoricas | p-value < 0.05 | Compara las frecuencias observadas en los datos actuales contra las frecuencias esperadas segun la distribucion de referencia |

#### Semaforo PSI

| Color | Rango PSI | Interpretacion |
|---|---|---|
| Verde | < 0.1 | Sin cambio significativo. El modelo opera dentro de parametros normales |
| Amarillo | 0.1 - 0.25 | Cambio moderado. Se recomienda aumentar frecuencia de monitoreo |
| Rojo | > 0.25 | Cambio significativo. Considerar reentrenamiento del modelo |

#### Visualizaciones

- Semaforo PSI por variable con indicadores de color
- Tabla de metricas de drift con colores por estado (verde = sin drift, rojo = drift detectado)
- Histogramas comparativos de distribucion historica vs actual (variables numericas)
- Graficos de barras para comparacion de categorias (variables categoricas)
- Barras de riesgo por variable (porcentaje de metricas con drift)

### Tab 3: Analisis Temporal

- Evolucion de la probabilidad promedio de prediccion a lo largo del tiempo
- Deteccion de cambios abruptos usando diferencias entre muestreos consecutivos
- Evolucion del estadistico KS por variable a traves del tiempo
- Requiere multiples muestreos para generar tendencias

### Tab 4: Recomendaciones

Genera mensajes automaticos basados en los resultados del analisis de drift:

- Si 0% de metricas con drift: modelo estable, sin accion requerida
- Si menos del 20%: drift leve, monitorear con mayor frecuencia
- Si menos del 50%: drift moderado, re-evaluar desempeno del modelo
- Si 50% o mas: drift significativo, se recomienda reentrenar el modelo
- Si PSI > 0.25 en alguna variable: alerta critica de cambio poblacional
- Sugerencias concretas de accion segun nivel de severidad

## Instrucciones de Ejecucion

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Entrenar modelos (solo si no existen los archivos .joblib)

```bash
cd src
python model_training.py
```

Esto genera en la carpeta `models/`:
- `xgboost.joblib` (modelo seleccionado con threshold optimizado)
- `preprocessor.joblib` (pipeline de preprocesamiento)
- Otros modelos evaluados (logistic_regression, decision_tree, random_forest, gradient_boosting)

### 3. Levantar la API

```bash
cd src
python -m uvicorn model_deploy:app --reload --port 8000
```

Verificar que funciona en: http://localhost:8000/health

### 4. Levantar el Dashboard de Monitoreo

En otra terminal:

```bash
cd src
streamlit run model_monitoring.py
```

Se abrira automaticamente en: http://localhost:8501

### Flujo de uso del monitoreo

1. Verificar que la API esta corriendo (el sidebar muestra el estado de conexion)
2. Configurar el tamano de muestra con el slider del sidebar
3. Hacer clic en "Generar nuevas predicciones" para simular la llegada de datos nuevos
4. Revisar las metricas en el Tab 1 (Metricas y Graficas)
5. Analizar el drift en el Tab 2 (Data Drift) - ver semaforo PSI, tabla de metricas, distribuciones
6. Repetir el muestreo varias veces para alimentar el Tab 3 (Analisis Temporal)
7. Consultar las recomendaciones automaticas en el Tab 4

## Proceso y Hallazgos

### Proceso de desarrollo

1. Limpieza y transformacion de datos: eliminacion de variables con data leakage, codificacion de categoricas, escalado de numericas
2. Entrenamiento de 5 modelos candidatos con validacion cruzada
3. Seleccion del mejor modelo (XGBoost) basado en AUC-ROC y optimizacion del threshold
4. Despliegue como API REST con FastAPI
5. Construccion del sistema de monitoreo con Streamlit

### Hallazgos principales

- El modelo XGBoost fue seleccionado como el de mejor desempeno entre los 5 candidatos evaluados
- Las variables financieras (puntaje_datacredito, salario_cliente, capital_prestado) son las mas relevantes para la prediccion
- Estas mismas variables son las mas susceptibles a drift por cambios macroeconomicos, lo que justifica el monitoreo continuo
- El threshold del modelo fue optimizado para balancear precision y recall segun el caso de negocio

## Conclusiones y Recomendaciones

1. El monitoreo continuo es fundamental porque el data drift es inevitable en produccion. Este sistema permite detectarlo antes de que afecte las decisiones de negocio.

2. Se recomienda reentrenar el modelo cuando el PSI supere 0.25 en variables clave como puntaje_datacredito o salario_cliente.

3. Las variables financieras son las mas criticas de monitorear por su sensibilidad a cambios macroeconomicos, regulatorios o estacionales.

4. La frecuencia recomendada de monitoreo en produccion es semanal, o ante eventos externos relevantes como cambios regulatorios o crisis economicas.

5. El archivo monitoring_log.csv mantiene un registro historico de todas las predicciones, permitiendo auditoria y trazabilidad completa del modelo en produccion.

## Requisitos tecnicos

- Python 3.11+
- pandas, numpy, scikit-learn, xgboost
- FastAPI, uvicorn, pydantic
- Streamlit, plotly
- scipy (metricas de drift)
- evidently (reporte opcional de drift)
- requests (comunicacion con API)

