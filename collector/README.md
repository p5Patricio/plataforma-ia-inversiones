# Collector

El collector descarga historicos OHLCV desde proveedores externos y los normaliza al formato que usa el pipeline de entrenamiento:

```text
timestamp,open,high,low,close,volume,ticker,source
```

## Proveedores iniciales

| Provider | Uso recomendado | Ejemplo de ticker |
|---|---|---|
| `yfinance` | Prototipo rapido de acciones, ETFs, indices y crypto | `AAPL`, `SPY`, `BTC-USD` |
| `stooq` | Historicos CSV gratuitos para acciones/indices/FX | `aapl.us`, `spy.us`, `^spx` |
| `binance` | Crypto OHLCV por exchange | `BTCUSDT`, `ETHUSDT` |

## Descargar CSV historico

```powershell
python -m collector.download_history --provider stooq --ticker aapl.us --start 2020-01-01 --out data/raw/aapl_stooq.csv
python -m collector.download_history --provider binance --ticker BTCUSDT --interval 1d --start 2020-01-01 --out data/raw/btcusdt_binance.csv
python -m collector.download_history --provider yfinance --ticker SPY --start 2020-01-01 --out data/raw/spy_yfinance.csv
```

Ese CSV puede alimentar el entrenamiento:

```powershell
python -m brain.train --prices-csv data/raw/aapl_stooq.csv --model-out models/aapl.joblib --metrics-out reports/aapl_metrics.json
```

## Cargar historico a Supabase

Configura `SUPABASE_URL` y `SUPABASE_KEY` en `.env`, luego ejecuta:

```powershell
python -m collector.load_history_to_supabase --provider stooq --ticker aapl.us --asset-ticker AAPL --asset-class stock --start 2020-01-01
python -m collector.load_history_to_supabase --provider binance --ticker BTCUSDT --asset-ticker BTCUSDT --asset-class crypto --start 2020-01-01
```

El comando:

1. descarga precios desde el proveedor;
2. normaliza OHLCV;
3. crea el activo si no existe;
4. hace upsert de precios por `asset_id,timestamp`.

## Job configurable de ingesta

Para cargar el universo inicial definido en `collector/main.py`:

```powershell
python -m collector.main
```

Tambien puedes pasar un archivo JSON con tu propio universo:

```json
[
  {
    "provider": "stooq",
    "ticker": "aapl.us",
    "asset_ticker": "AAPL",
    "name": "Apple Inc.",
    "asset_class": "stock",
    "interval": "1d",
    "start": "2020-01-01"
  },
  {
    "provider": "binance",
    "ticker": "BTCUSDT",
    "asset_ticker": "BTCUSDT",
    "name": "Bitcoin / Tether",
    "asset_class": "crypto",
    "interval": "1d",
    "start": "2020-01-01"
  }
]
```

```powershell
python -m collector.main --assets-file data/assets/universe.json
python -m collector.main --assets-file data/assets/universe.json --start 2021-01-01 --end 2024-12-31
```

## Integracion siguiente

El siguiente paso es crear tablas para `features_daily`, `labels_daily`, `predictions` y `model_runs`, de modo que la ingesta historica pueda alimentar entrenamiento, backtests y senales versionadas.
