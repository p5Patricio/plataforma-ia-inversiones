-- Feature rows generated from point-in-time market data.
create table if not exists features_daily (
  id bigint primary key generated always as identity,
  asset_id uuid references assets(id) on delete cascade,
  timestamp timestamptz not null,
  feature_set text not null,
  features jsonb not null,
  created_at timestamptz default now(),
  unique(asset_id, timestamp, feature_set)
);

create index if not exists features_daily_asset_timestamp_idx
  on features_daily(asset_id, timestamp);

-- Supervised labels generated from future outcomes.
create table if not exists labels_daily (
  id bigint primary key generated always as identity,
  asset_id uuid references assets(id) on delete cascade,
  timestamp timestamptz not null,
  label_method text not null,
  horizon integer not null,
  label text not null,
  outcome_return numeric,
  label_exit_timestamp timestamptz,
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz default now(),
  unique(asset_id, timestamp, label_method, horizon)
);

create index if not exists labels_daily_asset_timestamp_idx
  on labels_daily(asset_id, timestamp);

-- Training/evaluation runs for reproducibility.
create table if not exists model_runs (
  id uuid primary key default gen_random_uuid(),
  model_name text not null,
  model_version text not null,
  feature_set text not null,
  label_method text not null,
  horizon integer not null,
  train_start timestamptz,
  train_end timestamptz,
  params jsonb default '{}'::jsonb,
  metrics jsonb default '{}'::jsonb,
  artifact_uri text,
  created_at timestamptz default now(),
  unique(model_name, model_version)
);

-- Point-in-time predictions emitted by a model version.
create table if not exists predictions (
  id bigint primary key generated always as identity,
  asset_id uuid references assets(id) on delete cascade,
  model_run_id uuid references model_runs(id) on delete set null,
  timestamp timestamptz not null,
  action text not null,
  confidence numeric,
  expected_return numeric,
  expected_risk numeric,
  probabilities jsonb default '{}'::jsonb,
  metadata jsonb default '{}'::jsonb,
  created_at timestamptz default now(),
  unique(asset_id, model_run_id, timestamp)
);

create index if not exists predictions_asset_timestamp_idx
  on predictions(asset_id, timestamp);

-- Feedback view used for model monitoring and future retraining.
-- A prediction becomes evaluable once the matching label has been materialized.
create or replace view prediction_feedback as
select
  p.id as prediction_id,
  p.asset_id,
  p.model_run_id,
  p.timestamp,
  p.action as predicted_action,
  p.confidence,
  p.expected_return,
  p.expected_risk,
  p.probabilities,
  p.metadata,
  mr.model_name,
  mr.model_version,
  mr.feature_set,
  mr.label_method,
  mr.horizon,
  l.label as actual_label,
  l.outcome_return,
  case
    when l.label is null then null
    else p.action = l.label
  end as is_correct,
  p.created_at as prediction_created_at
from predictions p
join model_runs mr on mr.id = p.model_run_id
left join labels_daily l
  on l.asset_id = p.asset_id
 and l.timestamp = p.timestamp
 and l.label_method = mr.label_method
 and l.horizon = mr.horizon;

-- Backtest summaries for evaluated prediction streams.
create table if not exists backtests (
  id uuid primary key default gen_random_uuid(),
  model_run_id uuid references model_runs(id) on delete set null,
  asset_id uuid references assets(id) on delete cascade,
  name text not null,
  started_at timestamptz,
  ended_at timestamptz,
  params jsonb default '{}'::jsonb,
  metrics jsonb default '{}'::jsonb,
  created_at timestamptz default now()
);

create index if not exists backtests_model_asset_idx
  on backtests(model_run_id, asset_id);

-- Simulated trades produced by a backtest.
create table if not exists backtest_trades (
  id bigint primary key generated always as identity,
  backtest_id uuid references backtests(id) on delete cascade,
  asset_id uuid references assets(id) on delete cascade,
  timestamp timestamptz not null,
  action text not null,
  confidence numeric,
  gross_return numeric,
  net_return numeric,
  cost numeric,
  equity numeric,
  metadata jsonb default '{}'::jsonb
);

create index if not exists backtest_trades_backtest_timestamp_idx
  on backtest_trades(backtest_id, timestamp);

-- Optional risk policies that can be applied globally or per asset.
create table if not exists risk_limits (
  id bigint primary key generated always as identity,
  asset_id uuid references assets(id) on delete cascade,
  name text not null,
  max_position_size numeric not null default 0.10,
  min_confidence_to_trade numeric not null default 0.60,
  max_expected_risk numeric not null default 0.05,
  stop_loss numeric not null default 0.02,
  take_profit numeric not null default 0.04,
  allow_short boolean not null default true,
  is_active boolean not null default true,
  created_at timestamptz default now(),
  unique(asset_id, name)
);
