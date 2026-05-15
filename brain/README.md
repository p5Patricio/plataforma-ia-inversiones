# Brain

El modulo `brain` transforma precios historicos en datasets entrenables, etiquetas supervisadas y modelos de senales.

## Flujo actual

1. Cargar precios historicos con `collector`.
2. Materializar features y labels en Supabase.
3. Entrenar un modelo desde CSV o desde datasets exportados.
4. Comparar resultados con walk-forward validation.

## Materializar features y labels

Despues de cargar precios en Supabase:

```powershell
python -m brain.materialize_dataset --ticker AAPL --label-method triple_barrier --horizon 5
python -m brain.materialize_dataset --ticker BTCUSDT --feature-set technical_v1 --label-method fixed_horizon --horizon 5
```

Este comando:

1. busca el activo en Supabase;
2. descarga sus precios;
3. calcula features tecnicas point-in-time;
4. calcula labels supervisadas;
5. guarda resultados en `features_daily` y `labels_daily`.

## Entrenamiento desde CSV

```powershell
python -m brain.train --prices-csv data/raw/aapl_stooq.csv --model-out models/aapl.joblib --metrics-out reports/aapl_metrics.json
```

El entrenamiento usa validacion temporal walk-forward. No usa splits aleatorios.

## Entrenamiento desde Supabase

Despues de materializar features y labels:

```powershell
python -m brain.train_from_supabase --ticker AAPL --feature-set technical_v1 --label-method triple_barrier --horizon 5
```

Este comando:

1. lee `features_daily` y `labels_daily`;
2. reconstruye el dataset supervisado;
3. ejecuta walk-forward validation;
4. entrena el modelo final;
5. guarda el artefacto en `models/`;
6. registra la corrida en `model_runs`.

## Generar predicciones versionadas

Con un `model_run` registrado:

```powershell
python -m brain.predict_from_supabase --ticker AAPL --model-name baseline_hist_gradient_boosting --model-version 20260705120000
```

Este comando:

1. lee el `model_run`;
2. carga el artefacto `.joblib`;
3. toma las features materializadas mas recientes;
4. genera probabilidades por clase;
5. aplica un umbral minimo de confianza;
6. guarda la salida en `predictions`.

Las predicciones quedan preparadas para evaluarse contra labels futuros mediante la vista `prediction_feedback`. Ese feedback debe usarse en reentrenamientos o meta-modelos; no se debe tratar una prediccion pasada como verdad hasta que exista su resultado real.

### Risk engine

El comando de prediccion tambien aplica reglas de riesgo:

```powershell
python -m brain.predict_from_supabase --ticker AAPL --model-name baseline_hist_gradient_boosting --model-version 20260705120000 --min-confidence-to-trade 0.65 --max-position-size 0.08 --stop-loss 0.02 --take-profit 0.04
```

Las reglas pueden:

- convertir una senal debil a `HOLD`;
- bloquear shorts con `--no-short`;
- limitar el tamano de posicion;
- guardar `position_size`, `stop_loss`, `take_profit` y razones de bloqueo en `predictions.metadata`.

## Analizar feedback

Cuando ya existan labels para predicciones anteriores:

```powershell
python -m brain.feedback_report --ticker AAPL --model-name baseline_hist_gradient_boosting --model-version 20260705120000
python -m brain.feedback_report --model-name baseline_hist_gradient_boosting --model-version 20260705120000 --out reports/feedback_aapl.json
```

El reporte resume:

- predicciones evaluadas;
- accuracy;
- retorno medio y total;
- desempeno por accion predicha;
- desempeno por bucket de confianza.

Este reporte es la entrada segura para decidir si un modelo debe reentrenarse, ajustar su umbral de confianza o alimentar un meta-modelo.

## Backtesting de predicciones

Para probar si las senales hubieran ganado dinero despues de costos:

```powershell
python -m brain.backtest_predictions --ticker AAPL --model-name baseline_hist_gradient_boosting --model-version 20260705120000
python -m brain.backtest_predictions --ticker AAPL --model-name baseline_hist_gradient_boosting --model-version 20260705120000 --fee-bps 5 --slippage-bps 5 --persist
```

El backtest usa predicciones ya evaluadas desde `prediction_feedback`. Calcula:

- retorno neto;
- equity final;
- max drawdown;
- win rate;
- profit factor;
- exposicion;
- costos estimados.

Con `--persist`, guarda el resumen en `backtests` y las operaciones simuladas en `backtest_trades`.
