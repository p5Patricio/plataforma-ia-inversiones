-- Tabla de Activos (Stocks, Crypto, etc.)
create table assets (
  id uuid primary key default gen_random_uuid(),
  ticker text unique not null,
  name text,
  asset_class text -- 'stock', 'crypto', 'etf'
);

-- Tabla de Precios Históricos (OHLCV)
create table prices (
  id bigint primary key generated always as identity,
  asset_id uuid references assets(id) on delete cascade,
  timestamp timestamptz not null,
  open numeric,
  high numeric,
  low numeric,
  close numeric,
  volume bigint,
  unique(asset_id, timestamp)
);

-- Tabla de Señales de IA
create table signals (
  id bigint primary key generated always as identity,
  asset_id uuid references assets(id) on delete cascade,
  timestamp timestamptz default now(),
  signal_type text, -- 'buy', 'sell', 'hold'
  confidence numeric,
  metadata jsonb -- para guardar por qué la IA tomó la decisión
);
