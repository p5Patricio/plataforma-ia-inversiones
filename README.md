# IA Inversiones

![IA Inversiones](ui/public/brand/ia-inversiones-logo.png)

Plataforma experimental para investigacion, entrenamiento y evaluacion de modelos de decision de inversion. El objetivo es convertir datos historicos de mercado en senales auditables de **comprar**, **vender** o **mantener**, siempre acompanadas por confianza, riesgo, probabilidades y trazabilidad del modelo.

> Este proyecto no es asesoria financiera. Las senales deben validarse con backtesting, gestion de riesgo y supervision humana antes de cualquier uso real.

## Estado Actual

| Area | Estado |
| --- | --- |
| Frontend | Dashboard React con activos, grafico, senal, riesgo, probabilidades, backtests e historial de predicciones. |
| API | FastAPI con endpoints para activos, precios, analisis, backtests e historial de predicciones. |
| Datos | Supabase como fuente principal; modo demo local cuando Supabase no esta disponible. |
| ML | Pipeline base para features, labels, entrenamiento, inferencia, feedback y backtesting. |
| Calidad | Suite de pruebas para API, collector, repositorio Supabase y pipeline de modelo. |

## Experiencia

La interfaz esta pensada como una consola operativa, no como landing page. El usuario ve primero:

- Activo seleccionado y clase de activo.
- Senal actual: `BUY`, `SELL` o `HOLD`.
- Confianza y horizonte.
- Grafico historico de precio.
- Gestion de riesgo: posicion, stop, objetivo y bloqueos.
- Probabilidades por accion.
- Backtests persistidos por instrumento y version de modelo.
- Metadatos del modelo o indicador que genero la lectura.
- Historial reciente para auditar predicciones y feedback.

Cuando la API no puede conectarse a Supabase, la aplicacion muestra `Datos demo` para evitar confundir datos sinteticos con datos reales.

## Arquitectura

```mermaid
flowchart LR
  sources["Market data providers"] --> collector["collector"]
  collector --> supabase["Supabase"]
  supabase --> brain["brain: features, labels, training"]
  brain --> predictions["predictions + feedback"]
  predictions --> api["FastAPI"]
  supabase --> api
  api --> ui["React dashboard"]
```

## Estructura

| Ruta | Proposito |
| --- | --- |
| `api/` | API HTTP con FastAPI. |
| `brain/` | Features, labeling, entrenamiento, inferencia, feedback y backtesting. |
| `collector/` | Descarga y carga de historicos hacia Supabase. |
| `supabase/migrations/` | Esquema SQL para datos, modelos, predicciones y feedback. |
| `ui/` | Frontend React + Vite + Tailwind. |
| `tests/` | Pruebas automatizadas del sistema. |
| `INVESTIGACION_MODELO_PREDICTIVO.md` | Guia de investigacion y hoja de ruta tecnica. |

## Configuracion

1. Crea una copia local de variables:

```bash
cp .env.example .env
```

2. Completa tus credenciales:

```env
APP_ENV=development
ALLOW_DEMO_FALLBACK=true
API_CORS_ORIGINS=*
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_KEY=tu-clave-server-side-local
VITE_API_BASE_URL=http://localhost:8000/api
VITE_SUPABASE_URL=https://tu-proyecto.supabase.co
VITE_SUPABASE_ANON_KEY=tu-clave-publica-anon
```

`SUPABASE_KEY` se usa solo en backend, ingestion, entrenamiento e inferencia. No debe exponerse en el frontend ni subirse al repositorio; para ambientes con RLS activado usa una clave server-side creada para el pipeline.

El frontend esta configurado para leer las variables `VITE_*` desde este `.env` de la raiz del repositorio.

Variables de entorno principales:

| Variable | Uso |
| --- | --- |
| `APP_ENV` | Entorno de ejecucion: `development`, `staging`, `production` o `test`. |
| `ALLOW_DEMO_FALLBACK` | Permite servir datos demo si Supabase no esta disponible. Por defecto es `true` fuera de produccion y `false` en `production`. |
| `API_CORS_ORIGINS` | Lista separada por comas de origenes permitidos por la API. |
| `VITE_API_BASE_URL` | URL base que usa el frontend para llamar a la API. |
| `VITE_SUPABASE_URL` | URL publica del proyecto Supabase usada por Supabase Auth en el frontend. |
| `VITE_SUPABASE_ANON_KEY` | Clave anon/public de Supabase para login del usuario; no uses una service role key aqui. |

3. Instala dependencias:

```bash
pip install -r requirements.txt
cd ui
npm install
```

## Ejecucion Local

API:

```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd ui
npm run dev -- --host 127.0.0.1 --port 5173
```

Abre [http://127.0.0.1:5173](http://127.0.0.1:5173).

## Verificacion

Backend y pipeline:

```bash
pytest tests
```

Frontend:

```bash
cd ui
npm run lint
npm run build
```

Conexion con Supabase:

```bash
python -c "from collector.supabase_repository import SupabaseConfig, SupabaseRepository; r=SupabaseRepository(SupabaseConfig.from_env()); print(len(r.get_assets()))"
```

Esquema ML en Supabase:

```bash
python -m collector.schema_check
```

Si falta alguna relacion, aplica las migraciones pendientes de `supabase/migrations/` desde Supabase SQL Editor o Supabase CLI y vuelve a ejecutar el chequeo.

Si falla DNS o red en desarrollo, la API activa el modo demo local para que el dashboard siga siendo navegable. En produccion, deja `ALLOW_DEMO_FALLBACK=false` para que los fallos de datos se reporten como errores reales en lugar de mostrarse como lecturas sinteticas.

## Flujo de Trabajo del Modelo

1. Descargar historicos de mercado por instrumento.
2. Cargar precios normalizados a Supabase.
3. Materializar features tecnicos y labels.
4. Entrenar modelos candidatos con validacion walk-forward.
5. Evaluar out-of-sample con backtesting, baselines y barrido de umbrales.
6. Guardar `model_runs`, predicciones y metadata.
7. Evaluar feedback de predicciones previas.
8. Servir la decision en la API con riesgo y trazabilidad.

## Paper Trading

La API puede simular una cuenta de paper trading con las predicciones ya guardadas y precios observados:

```bash
curl "http://127.0.0.1:8000/api/paper-trading/BTC-USD?initial_capital=10000&fee_bps=5&slippage_bps=5"
```

La simulacion mantiene posicion con `HOLD`, abre/ajusta long con `BUY`, abre/ajusta short con `SELL` cuando esta permitido, y aplica costos solo cuando cambia la exposicion. Devuelve metricas como equity final, retorno total, drawdown, operaciones ejecutadas, exposicion promedio y posicion abierta. El dashboard muestra estas metricas junto con la curva de equity, marcadores de operaciones y ultimas senales simuladas para revisar rapidamente como se habria comportado la estrategia.

Para guardar una simulacion y compararla despues, aplica `supabase/migrations/20260707000300_paper_trading_runs.sql` y ejecuta:

```bash
curl "http://127.0.0.1:8000/api/paper-trading/BTC-USD?persist=true&initial_capital=10000&fee_bps=5&slippage_bps=5"
curl "http://127.0.0.1:8000/api/paper-trading-runs/BTC-USD?limit=10"
```

En el dashboard, el panel de paper trading permite guardar la corrida actual y comparar las corridas persistidas por retorno, drawdown, trades, equity y modelo.

Para persistir corridas periodicas por ticker desde un scheduler:

```bash
python -m brain.run_paper_trading_job --tickers BTC-USD,AAPL --out reports/paper_trading_job.json
```

## Monitoreo de Predicciones

La API expone calidad historica por activo con accuracy, confianza media y retorno realizado:

```bash
curl "http://127.0.0.1:8000/api/feedback/BTC-USD?limit=250"
```

El dashboard muestra este resumen en `Calidad del modelo` para detectar degradacion, sesgos por accion y necesidad de reentrenamiento.

## Salud Operativa

Para revisar API, Supabase y esquema requerido:

```bash
curl "http://127.0.0.1:8000/api/health"
```

El dashboard muestra el estado en el panel `Sistema`.

## Jobs Operativos

Para actualizar precios y materializar datasets:

```bash
python -m collector.run_market_data_job --assets-file config/assets.core.json --feature-sets technical_v2 --out reports/market_data_job.json
```

Para materializar un activo ya cargado en Supabase sin descargar precios:

```bash
python -m collector.run_market_data_job --skip-collection --tickers BTC-USD --feature-sets technical_v2 --out reports/market_data_job_btc.json
```

El job de datos:

- descarga precios para los activos configurados;
- guarda OHLCV normalizado en Supabase;
- materializa features y labels;
- reporta errores por activo sin detener todo el proceso, salvo que uses `--fail-fast`.

Para generar predicciones latest desde los modelos promovidos:

```bash
python -m brain.run_inference_job --out reports/inference_job_latest.json
```

El job:

- busca `model_runs` creados por promocion de candidatos;
- identifica su `target_ticker`;
- carga el artefacto `.joblib`;
- genera la prediccion mas reciente desde features materializadas;
- aplica reglas de riesgo;
- guarda la prediccion en Supabase.

El job de paper trading persiste simulaciones de las predicciones guardadas en `paper_trading_runs` y `paper_trading_events`, y alimenta el comparativo del dashboard.

Si el job corre fuera de tu maquina, el `model_run` debe apuntar a un artefacto remoto:

```bash
python -m brain.upload_model_artifact --model-name extra_trees --model-version promoted_smoke_20260706
```

Esto sube el `.joblib` a Supabase Storage y actualiza `artifact_uri` a `supabase://model-artifacts/...`, que tambien puede resolver GitHub Actions.

Puede filtrarse por version:

```bash
python -m brain.run_inference_job --model-name extra_trees --model-version promoted_smoke_20260706
```

### Scheduler Externo

El repositorio incluye `.github/workflows/operational-jobs.yml` para ejecutar jobs desde GitHub Actions:

- `schedule`: corre todos los dias a las 06:20 UTC y actualiza datos/features con `config/assets.core.json`.
- `workflow_dispatch`: permite lanzar `market_data`, `inference`, `paper_trading` o `full` manualmente.
- `tickers`: permite limitar una corrida a instrumentos concretos, por ejemplo `BTC-USD,AAPL`.
- `skip_collection`: materializa features y labels usando precios ya guardados.

Configura estos GitHub Secrets antes de activar el workflow:

```text
SUPABASE_URL
SUPABASE_KEY
```

El workflow corre con `APP_ENV=production` y `ALLOW_DEMO_FALLBACK=false`, por lo que falla rapido si Supabase o el esquema no estan disponibles. Los reportes JSON se suben como artifacts de la ejecucion, no se versionan en el repositorio.

## Perfiles de Riesgo

La API soporta perfiles de riesgo por usuario autenticado con Supabase Auth:

```bash
curl -H "Authorization: Bearer <access_token>" http://127.0.0.1:8000/api/risk-profile
```

Para guardar el perfil por defecto:

```bash
curl -X PUT http://127.0.0.1:8000/api/risk-profile \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"name":"default","max_position_size":0.05,"min_confidence_to_trade":0.7,"max_expected_risk":0.03,"stop_loss":0.02,"take_profit":0.05,"allow_short":false}'
```

Tambien puedes guardar perfiles por clase de activo o ticker. La prioridad al analizar un activo es `ticker > asset_class > default`:

```bash
curl -X PUT http://127.0.0.1:8000/api/risk-profile \
  -H "Authorization: Bearer <access_token>" \
  -H "Content-Type: application/json" \
  -d '{"name":"crypto","scope_type":"asset_class","scope_value":"crypto","max_position_size":0.03,"min_confidence_to_trade":0.75,"max_expected_risk":0.04,"stop_loss":0.02,"take_profit":0.06,"allow_short":false}'
```

Sin token, `GET /api/risk-profile` devuelve la politica conservadora por defecto. Para persistencia por usuario aplica `supabase/migrations/20260707000100_user_risk_profiles.sql` y luego `supabase/migrations/20260707000200_scoped_user_risk_profiles.sql`.

El frontend activa login y edicion del perfil cuando `VITE_SUPABASE_URL` y `VITE_SUPABASE_ANON_KEY` estan configuradas. Desde el panel `Perfil`, el usuario puede alternar entre editar el perfil global, el de la clase del activo seleccionado o el del ticker seleccionado.

Cuando el dashboard llama `GET /api/analysis/{ticker}` con un token de usuario, la API conserva la prediccion versionada del modelo pero recalcula la accion final, el tamano de posicion, stop, objetivo y bloqueos con el perfil de riesgo autenticado. Esto permite que dos usuarios vean la misma prediccion base con decisiones operativas distintas segun sus limites.

La interfaz muestra esa transparencia como `Modelo base -> decision final`, junto con el perfil aplicado y las razones de bloqueo cuando el motor de riesgo cambia o condiciona la senal.

## Seguridad Para Repos Publicos

- No publiques `.env`.
- Usa `.env.example` para documentar variables.
- Usa claves server-side solo en procesos privados de backend/pipeline, nunca en el frontend.
- Revisa que las credenciales de Supabase no queden en commits.
- Rota cualquier credencial que haya sido expuesta previamente.

## Roadmap

- Aplicar perfiles de riesgo de usuario durante inferencia personalizada.
