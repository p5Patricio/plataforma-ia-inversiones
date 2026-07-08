-- Harden public market/model tables with RLS.
-- Backend jobs use server-side credentials; clients only get read access to
-- reference data and model outputs, never direct write access.

do $$
declare
  table_name text;
begin
  foreach table_name in array array[
    'assets',
    'prices',
    'signals',
    'features_daily',
    'labels_daily',
    'model_runs',
    'predictions',
    'backtests',
    'backtest_trades',
    'risk_limits',
    'paper_trading_runs',
    'paper_trading_events',
    'news_events'
  ]
  loop
    if to_regclass(format('public.%I', table_name)) is not null then
      execute format('alter table public.%I enable row level security', table_name);
    end if;
  end loop;
end $$;

do $$
declare
  table_name text;
  policy_name text;
begin
  foreach table_name in array array[
    'assets',
    'prices',
    'signals',
    'features_daily',
    'labels_daily',
    'model_runs',
    'predictions',
    'backtests',
    'backtest_trades',
    'risk_limits',
    'paper_trading_runs',
    'paper_trading_events',
    'news_events'
  ]
  loop
    if to_regclass(format('public.%I', table_name)) is not null then
      policy_name := format('%s_public_read', table_name);
      execute format('drop policy if exists %I on public.%I', policy_name, table_name);
      execute format(
        'create policy %I on public.%I for select to anon, authenticated using (true)',
        policy_name,
        table_name
      );
    end if;
  end loop;
end $$;

do $$
begin
  if to_regclass('public.prediction_feedback') is not null then
    alter view public.prediction_feedback set (security_invoker = true);
  end if;
exception
  when undefined_object or feature_not_supported then
    null;
end $$;
