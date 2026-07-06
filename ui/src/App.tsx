import { useCallback, useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  CircleDollarSign,
  Clock3,
  Gauge,
  History,
  MinusCircle,
  RefreshCcw,
  ShieldCheck,
  TrendingDown,
  TrendingUp,
  type LucideIcon,
} from 'lucide-react';
import { FinancialChart, type PricePoint } from './components/FinancialChart';

const API_BASE_URL = 'http://localhost:8000/api';

type Signal = 'BUY' | 'SELL' | 'HOLD' | string;

interface Asset {
  id: string;
  ticker: string;
  name?: string;
  asset_class?: string;
}

interface RiskMetadata {
  position_size?: number;
  stop_loss?: number | null;
  take_profit?: number | null;
  blocked_reasons?: string[];
}

interface ModelMetadata {
  name?: string;
  version?: string;
  run_id?: string;
  feature_set?: string;
  label_method?: string;
  horizon?: number;
}

interface FeedbackMetadata {
  actual_label?: string | null;
  is_correct?: boolean | null;
  outcome_return?: number | null;
}

interface Analysis {
  signal: Signal;
  confidence: number;
  reason?: string;
  reasons?: string[];
  probabilities?: Record<string, number>;
  model?: ModelMetadata;
  risk?: RiskMetadata;
  feedback?: FeedbackMetadata;
  indicators?: {
    rsi?: number | null;
    close?: number | null;
    sma_20?: number | null;
  };
  prediction_timestamp?: string;
  expected_return?: number | null;
  expected_risk?: number | null;
}

interface AnalysisResponse {
  ticker: string;
  timestamp: string;
  source?: 'prediction' | 'fallback_indicators' | 'demo_indicators' | string;
  analysis: Analysis;
}

interface PredictionAuditRow {
  prediction_id?: number;
  timestamp?: string;
  created_at?: string;
  action: Signal;
  confidence?: number | null;
  probabilities?: Record<string, number>;
  expected_return?: number | null;
  expected_risk?: number | null;
  model?: ModelMetadata;
  risk?: RiskMetadata;
  feedback?: FeedbackMetadata;
}

interface BacktestSummaryRow {
  id?: string;
  name?: string;
  started_at?: string | null;
  ended_at?: string | null;
  created_at?: string | null;
  metrics?: {
    total_return?: number | null;
    max_drawdown?: number | null;
    profit_factor?: number | null;
    active_trade_count?: number | null;
    trade_count?: number | null;
    win_rate?: number | null;
    exposure?: number | null;
    final_equity?: number | null;
  };
  model?: ModelMetadata;
}

function App() {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [selectedTicker, setSelectedTicker] = useState('');
  const [prices, setPrices] = useState<PricePoint[]>([]);
  const [analysisResponse, setAnalysisResponse] = useState<AnalysisResponse | null>(null);
  const [predictionHistory, setPredictionHistory] = useState<PredictionAuditRow[]>([]);
  const [backtests, setBacktests] = useState<BacktestSummaryRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAssets = useCallback(async () => {
    try {
      const response = await axios.get<Asset[]>(`${API_BASE_URL}/assets`);
      setAssets(response.data);
      setSelectedTicker((currentTicker) => currentTicker || response.data[0]?.ticker || '');
    } catch {
      setError('No se pudieron cargar los activos.');
    }
  }, []);

  const fetchData = useCallback(async (ticker: string) => {
    setLoading(true);
    setError(null);
    try {
      const [pricesResponse, analysisResponse, historyResponse, backtestsResponse] = await Promise.all([
        axios.get<PricePoint[]>(`${API_BASE_URL}/prices/${ticker}?limit=240`),
        axios.get<AnalysisResponse>(`${API_BASE_URL}/analysis/${ticker}`),
        axios
          .get<PredictionAuditRow[]>(`${API_BASE_URL}/predictions/${ticker}?limit=8`)
          .catch(() => ({ data: [] as PredictionAuditRow[] })),
        axios
          .get<BacktestSummaryRow[]>(`${API_BASE_URL}/backtests/${ticker}?limit=5`)
          .catch(() => ({ data: [] as BacktestSummaryRow[] })),
      ]);
      setPrices(pricesResponse.data);
      setAnalysisResponse(analysisResponse.data);
      setPredictionHistory(historyResponse.data);
      setBacktests(backtestsResponse.data);
    } catch {
      setError('No se pudo actualizar la señal.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    let disposed = false;
    queueMicrotask(() => {
      if (!disposed) {
        void fetchAssets();
      }
    });

    return () => {
      disposed = true;
    };
  }, [fetchAssets]);

  useEffect(() => {
    if (!selectedTicker) {
      return;
    }

    let disposed = false;
    queueMicrotask(() => {
      if (!disposed) {
        void fetchData(selectedTicker);
      }
    });

    return () => {
      disposed = true;
    };
  }, [fetchData, selectedTicker]);

  const selectedAsset = useMemo(
    () => assets.find((asset) => asset.ticker === selectedTicker),
    [assets, selectedTicker],
  );

  const analysis = analysisResponse?.analysis ?? null;

  return (
    <div className="min-h-screen bg-[#111312] text-zinc-100">
      <header className="border-b border-white/10 bg-[#171918]">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-4 py-5 md:flex-row md:items-center md:justify-between md:px-6">
          <div className="flex items-center gap-3">
            <img
              src="/brand/ia-inversiones-logo.png"
              alt="IA Inversiones"
              className="h-12 w-12 rounded-lg border border-amber-200/20 bg-zinc-50 object-cover"
            />
            <div>
              <h1 className="text-xl font-semibold tracking-normal text-zinc-50">IA Inversiones</h1>
              <p className="text-sm text-zinc-400">Decisiones de mercado con riesgo visible</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <StatusPill source={analysisResponse?.source} loading={loading} />
            <button
              type="button"
              onClick={() => selectedTicker && fetchData(selectedTicker)}
              className="inline-flex h-10 w-10 items-center justify-center rounded-lg border border-white/10 bg-white/[0.04] text-zinc-200 transition hover:bg-white/[0.08] focus:outline-none focus:ring-2 focus:ring-emerald-300/50"
              aria-label="Actualizar datos"
              title="Actualizar"
            >
              <RefreshCcw aria-hidden="true" className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl grid-cols-1 gap-5 px-4 py-5 md:px-6 lg:grid-cols-[280px_1fr]">
        <aside className="space-y-4">
          <section className="rounded-lg border border-white/10 bg-[#181b1a] p-3">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-medium text-zinc-200">Activos</h2>
              <span className="text-xs text-zinc-500">{assets.length}</span>
            </div>
            <div className="space-y-2">
              {assets.map((asset) => (
                <button
                  key={asset.id}
                  type="button"
                  onClick={() => setSelectedTicker(asset.ticker)}
                  className={`w-full rounded-lg border px-3 py-3 text-left transition focus:outline-none focus:ring-2 focus:ring-emerald-300/40 ${
                    selectedTicker === asset.ticker
                      ? 'border-emerald-300/40 bg-emerald-300/10'
                      : 'border-white/10 bg-white/[0.03] hover:bg-white/[0.06]'
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span className="font-medium text-zinc-50">{asset.ticker}</span>
                    <span className="rounded-md bg-zinc-900 px-2 py-1 text-[11px] uppercase text-zinc-400">
                      {asset.asset_class ?? 'asset'}
                    </span>
                  </div>
                  <p className="mt-1 truncate text-sm text-zinc-400">{asset.name ?? 'Sin nombre'}</p>
                </button>
              ))}
            </div>
          </section>

          <section className="rounded-lg border border-white/10 bg-[#181b1a] p-4">
            <div className="flex items-center gap-2 text-sm font-medium text-zinc-200">
              <Clock3 aria-hidden="true" className="h-4 w-4 text-amber-300" />
              Última lectura
            </div>
            <p className="mt-3 text-sm text-zinc-400">
              {analysisResponse?.timestamp ? formatDateTime(analysisResponse.timestamp) : 'Sin datos'}
            </p>
            {error && (
              <div className="mt-3 rounded-lg border border-red-400/30 bg-red-400/10 px-3 py-2 text-sm text-red-200">
                {error}
              </div>
            )}
          </section>
        </aside>

        <section className="space-y-5">
          <DecisionHeader asset={selectedAsset} analysis={analysis} />

          <div className="grid grid-cols-1 gap-5 xl:grid-cols-[1.4fr_0.8fr]">
            <section className="rounded-lg border border-white/10 bg-[#181b1a] p-4">
              <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className="text-base font-medium text-zinc-100">Precio</h2>
                  <p className="text-sm text-zinc-400">{selectedTicker || 'Selecciona un activo'}</p>
                </div>
                <PriceSnapshot prices={prices} />
              </div>
              {prices.length > 0 ? (
                <FinancialChart data={prices} />
              ) : (
                <div className="flex h-[420px] items-center justify-center rounded-lg border border-dashed border-white/10 text-sm text-zinc-500">
                  Sin histórico disponible
                </div>
              )}
            </section>

            <div className="space-y-5">
              <RiskPanel analysis={analysis} />
              <ProbabilityPanel probabilities={analysis?.probabilities} />
              <ModelPanel analysis={analysis} />
            </div>
          </div>

          <BacktestPanel rows={backtests} />
          <PredictionHistoryPanel rows={predictionHistory} />
        </section>
      </main>
    </div>
  );
}

function DecisionHeader({ asset, analysis }: { asset?: Asset; analysis: Analysis | null }) {
  const signal = analysis?.signal ?? 'HOLD';
  const tone = signalTone(signal);
  const reasons = analysis?.reasons ?? (analysis?.reason ? [analysis.reason] : []);

  return (
    <section className={`rounded-lg border p-5 ${tone.surface}`}>
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_220px]">
        <div>
          <div className="mb-4 flex items-center gap-3">
            <div className={`flex h-12 w-12 items-center justify-center rounded-lg ${tone.iconBg}`}>
              <SignalIcon signal={signal} className={`h-6 w-6 ${tone.icon}`} />
            </div>
            <div>
              <p className="text-sm text-zinc-400">{asset?.name ?? 'Activo seleccionado'}</p>
              <h2 className="text-2xl font-semibold tracking-normal text-zinc-50">{asset?.ticker ?? '...'}</h2>
            </div>
          </div>

          <div className="flex flex-wrap items-end gap-4">
            <div>
              <p className="text-sm text-zinc-400">Decisión</p>
              <p className={`text-4xl font-semibold tracking-normal ${tone.text}`}>{signal}</p>
            </div>
            <MetricInline label="Confianza" value={formatPercent(analysis?.confidence)} />
            <MetricInline label="Horizonte" value={analysis?.model?.horizon ? `${analysis.model.horizon}d` : 'N/D'} />
          </div>

          {reasons.length > 0 && (
            <div className="mt-4 flex flex-wrap gap-2">
              {reasons.map((reason) => (
                <span key={reason} className="rounded-md border border-white/10 bg-black/20 px-2 py-1 text-xs text-zinc-300">
                  {reason}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 gap-3 lg:grid-cols-1">
          <MetricBox icon={ShieldCheck} label="Posición" value={formatPercent(analysis?.risk?.position_size)} />
          <MetricBox icon={Gauge} label="Riesgo esperado" value={formatPercent(analysis?.expected_risk ?? null)} />
        </div>
      </div>
    </section>
  );
}

function RiskPanel({ analysis }: { analysis: Analysis | null }) {
  const risk = analysis?.risk;
  const blocked = risk?.blocked_reasons ?? [];

  return (
    <section className="rounded-lg border border-white/10 bg-[#181b1a] p-4">
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldCheck aria-hidden="true" className="h-4 w-4 text-emerald-300" />
          <h2 className="text-sm font-medium text-zinc-100">Riesgo</h2>
        </div>
        {blocked.length > 0 ? (
          <span className="rounded-md bg-amber-300/10 px-2 py-1 text-xs text-amber-200">Bloqueada</span>
        ) : (
          <span className="rounded-md bg-emerald-300/10 px-2 py-1 text-xs text-emerald-200">Activa</span>
        )}
      </div>

      <div className="grid grid-cols-3 gap-3">
        <SmallMetric label="Tamaño" value={formatPercent(risk?.position_size)} />
        <SmallMetric label="Stop" value={formatPercent(risk?.stop_loss)} />
        <SmallMetric label="Objetivo" value={formatPercent(risk?.take_profit)} />
      </div>

      {blocked.length > 0 && (
        <div className="mt-4 space-y-2">
          {blocked.map((reason) => (
            <div key={reason} className="flex items-center gap-2 rounded-lg border border-amber-300/20 bg-amber-300/10 px-3 py-2 text-sm text-amber-100">
              <AlertTriangle aria-hidden="true" className="h-4 w-4" />
              {humanizeReason(reason)}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function ProbabilityPanel({ probabilities }: { probabilities?: Record<string, number> }) {
  const entries = Object.entries(probabilities ?? { BUY: 0, HOLD: 0, SELL: 0 });

  return (
    <section className="rounded-lg border border-white/10 bg-[#181b1a] p-4">
      <div className="mb-4 flex items-center gap-2">
        <BarChart3 aria-hidden="true" className="h-4 w-4 text-sky-300" />
        <h2 className="text-sm font-medium text-zinc-100">Probabilidades</h2>
      </div>
      <div className="space-y-3">
        {entries.map(([label, value]) => (
          <div key={label}>
            <div className="mb-1 flex items-center justify-between text-sm">
              <span className="text-zinc-300">{label}</span>
              <span className="font-medium text-zinc-100">{formatPercent(value)}</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-zinc-800">
              <div className={`h-full ${probabilityColor(label)}`} style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function ModelPanel({ analysis }: { analysis: Analysis | null }) {
  const model = analysis?.model;
  const feedback = analysis?.feedback;

  return (
    <section className="rounded-lg border border-white/10 bg-[#181b1a] p-4">
      <div className="mb-4 flex items-center gap-2">
        <Brain aria-hidden="true" className="h-4 w-4 text-violet-300" />
        <h2 className="text-sm font-medium text-zinc-100">Modelo</h2>
      </div>
      <div className="space-y-3 text-sm">
        <InfoRow label="Nombre" value={model?.name ?? 'Indicadores'} />
        <InfoRow label="Versión" value={model?.version ?? 'Fallback'} />
        <InfoRow label="Feature set" value={model?.feature_set ?? 'N/D'} />
        <InfoRow label="Label" value={model?.label_method ?? 'N/D'} />
        <InfoRow label="Resultado" value={feedbackLabel(feedback)} />
      </div>
    </section>
  );
}

function BacktestPanel({ rows }: { rows: BacktestSummaryRow[] }) {
  const latest = rows[0];

  return (
    <section className="rounded-lg border border-white/10 bg-[#181b1a] p-4">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <BarChart3 aria-hidden="true" className="h-4 w-4 text-sky-300" />
          <h2 className="text-sm font-medium text-zinc-100">Backtests</h2>
        </div>
        <span className="text-xs text-zinc-500">{rows.length}</span>
      </div>

      {!latest ? (
        <div className="rounded-lg border border-dashed border-white/10 px-3 py-6 text-center text-sm text-zinc-500">
          Sin backtests persistidos
        </div>
      ) : (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <SmallMetric label="Retorno" value={formatPercent(latest.metrics?.total_return)} />
            <SmallMetric label="Drawdown" value={formatPercent(latest.metrics?.max_drawdown)} />
            <SmallMetric label="Profit factor" value={formatNumber(latest.metrics?.profit_factor)} />
            <SmallMetric label="Trades" value={formatCount(latest.metrics?.active_trade_count)} />
          </div>

          <div className="overflow-hidden rounded-lg border border-white/10">
            <div className="grid grid-cols-[1fr_88px_88px_72px] gap-3 border-b border-white/10 bg-black/20 px-3 py-2 text-xs text-zinc-500 md:grid-cols-[1fr_100px_100px_92px_96px]">
              <span>Modelo</span>
              <span>Retorno</span>
              <span>Drawdown</span>
              <span>PF</span>
              <span className="hidden md:block">Fecha</span>
            </div>
            <div className="divide-y divide-white/10">
              {rows.map((row) => (
                <div
                  key={row.id ?? row.name}
                  className="grid grid-cols-[1fr_88px_88px_72px] gap-3 px-3 py-3 text-sm md:grid-cols-[1fr_100px_100px_92px_96px]"
                >
                  <span className="min-w-0 truncate text-zinc-200">
                    {row.model?.name ?? 'Modelo'}:{row.model?.version ?? row.name ?? 'N/D'}
                  </span>
                  <span className={metricTone(row.metrics?.total_return)}>{formatPercent(row.metrics?.total_return)}</span>
                  <span className="text-zinc-300">{formatPercent(row.metrics?.max_drawdown)}</span>
                  <span className="text-zinc-300">{formatNumber(row.metrics?.profit_factor)}</span>
                  <span className="hidden text-zinc-400 md:block">{row.created_at ? formatShortDate(row.created_at) : 'N/D'}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function PredictionHistoryPanel({ rows }: { rows: PredictionAuditRow[] }) {
  return (
    <section className="rounded-lg border border-white/10 bg-[#181b1a] p-4">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <History aria-hidden="true" className="h-4 w-4 text-emerald-300" />
          <h2 className="text-sm font-medium text-zinc-100">Auditoria</h2>
        </div>
        <span className="text-xs text-zinc-500">{rows.length}</span>
      </div>

      {rows.length === 0 ? (
        <div className="rounded-lg border border-dashed border-white/10 px-3 py-6 text-center text-sm text-zinc-500">
          Sin predicciones historicas
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border border-white/10">
          <div className="grid grid-cols-[88px_1fr_88px_96px] gap-3 border-b border-white/10 bg-black/20 px-3 py-2 text-xs text-zinc-500 md:grid-cols-[116px_92px_1fr_96px_112px_120px]">
            <span>Fecha</span>
            <span className="hidden md:block">Accion</span>
            <span>Modelo</span>
            <span>Confianza</span>
            <span className="hidden md:block">Resultado</span>
            <span className="hidden md:block">Riesgo</span>
          </div>
          <div className="divide-y divide-white/10">
            {rows.map((row, index) => {
              const blocked = row.risk?.blocked_reasons ?? [];
              return (
                <div
                  key={`${row.prediction_id ?? row.timestamp ?? index}`}
                  className="grid grid-cols-[88px_1fr_88px_96px] gap-3 px-3 py-3 text-sm md:grid-cols-[116px_92px_1fr_96px_112px_120px]"
                >
                  <span className="text-zinc-400">{row.timestamp ? formatShortDate(row.timestamp) : 'N/D'}</span>
                  <span className={`hidden font-medium md:block ${signalTone(row.action).text}`}>{row.action}</span>
                  <span className="min-w-0 truncate text-zinc-200">
                    {row.model?.name ?? 'Modelo'}:{row.model?.version ?? 'N/D'}
                  </span>
                  <span className="text-zinc-200">{formatPercent(row.confidence)}</span>
                  <span className="hidden text-zinc-300 md:block">{feedbackLabel(row.feedback)}</span>
                  <span className="hidden truncate text-zinc-400 md:block">
                    {blocked.length > 0 ? blocked.map(humanizeReason).join(', ') : formatPercent(row.risk?.position_size)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}

function PriceSnapshot({ prices }: { prices: PricePoint[] }) {
  const latest = prices[0];
  if (!latest) {
    return <span className="text-sm text-zinc-500">Sin precio</span>;
  }

  return (
    <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-black/20 px-3 py-2">
      <CircleDollarSign aria-hidden="true" className="h-4 w-4 text-emerald-300" />
      <span className="text-sm font-medium text-zinc-100">{formatCurrency(Number(latest.close))}</span>
    </div>
  );
}

function StatusPill({ source, loading }: { source?: string; loading: boolean }) {
  if (loading) {
    return (
      <span className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-zinc-300">
        <Activity aria-hidden="true" className="h-4 w-4 animate-pulse text-sky-300" />
        Actualizando
      </span>
    );
  }

  const isPrediction = source === 'prediction';
  const isDemo = source === 'demo_indicators';
  return (
    <span className="inline-flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-zinc-300">
      {isPrediction ? (
        <CheckCircle2 aria-hidden="true" className="h-4 w-4 text-emerald-300" />
      ) : isDemo ? (
        <AlertTriangle aria-hidden="true" className="h-4 w-4 text-amber-300" />
      ) : (
        <MinusCircle aria-hidden="true" className="h-4 w-4 text-amber-300" />
      )}
      {isPrediction ? 'Modelo activo' : isDemo ? 'Datos demo' : 'Indicadores'}
    </span>
  );
}

function MetricInline({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-sm text-zinc-400">{label}</p>
      <p className="text-lg font-medium text-zinc-100">{value}</p>
    </div>
  );
}

function MetricBox({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/20 p-3">
      <Icon aria-hidden="true" className="mb-2 h-4 w-4 text-zinc-400" />
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="text-lg font-medium text-zinc-100">{value}</p>
    </div>
  );
}

function SmallMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/20 p-3">
      <p className="text-xs text-zinc-500">{label}</p>
      <p className="mt-1 text-sm font-medium text-zinc-100">{value}</p>
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-white/5 pb-2 last:border-b-0 last:pb-0">
      <span className="text-zinc-500">{label}</span>
      <span className="truncate text-right text-zinc-200">{value}</span>
    </div>
  );
}

function signalTone(signal: Signal) {
  if (signal === 'BUY') {
    return {
      surface: 'border-emerald-300/25 bg-emerald-300/[0.07]',
      text: 'text-emerald-300',
      icon: 'text-emerald-200',
      iconBg: 'bg-emerald-300/15',
    };
  }
  if (signal === 'SELL') {
    return {
      surface: 'border-red-300/25 bg-red-300/[0.07]',
      text: 'text-red-300',
      icon: 'text-red-200',
      iconBg: 'bg-red-300/15',
    };
  }
  return {
    surface: 'border-amber-300/25 bg-amber-300/[0.07]',
    text: 'text-amber-200',
    icon: 'text-amber-200',
    iconBg: 'bg-amber-300/15',
  };
}

function SignalIcon({ signal, className }: { signal: Signal; className: string }) {
  if (signal === 'BUY') return <TrendingUp aria-hidden="true" className={className} />;
  if (signal === 'SELL') return <TrendingDown aria-hidden="true" className={className} />;
  return <MinusCircle aria-hidden="true" className={className} />;
}

function probabilityColor(label: string) {
  if (label === 'BUY') return 'bg-emerald-300';
  if (label === 'SELL') return 'bg-red-300';
  return 'bg-amber-300';
}

function formatPercent(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/D';
  return `${(value * 100).toFixed(0)}%`;
}

function formatNumber(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/D';
  return value.toFixed(2);
}

function formatCount(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/D';
  return String(Math.round(value));
}

function metricTone(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'text-zinc-300';
  if (value > 0) return 'text-emerald-300';
  if (value < 0) return 'text-red-300';
  return 'text-zinc-300';
}

function formatCurrency(value: number) {
  return new Intl.NumberFormat('es-MX', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(value);
}

function formatDateTime(value: string) {
  return new Intl.DateTimeFormat('es-MX', {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(new Date(value));
}

function formatShortDate(value: string) {
  return new Intl.DateTimeFormat('es-MX', {
    month: 'short',
    day: 'numeric',
  }).format(new Date(value));
}

function humanizeReason(reason: string) {
  const map: Record<string, string> = {
    confidence_below_trade_threshold: 'Confianza insuficiente',
    short_disabled: 'Ventas en corto desactivadas',
    expected_risk_above_limit: 'Riesgo esperado sobre el límite',
  };
  return map[reason] ?? reason.replaceAll('_', ' ');
}

function feedbackLabel(feedback?: FeedbackMetadata) {
  if (!feedback || feedback.actual_label === undefined || feedback.actual_label === null) return 'Pendiente';
  if (feedback.is_correct === true) return `Acertó (${feedback.actual_label})`;
  if (feedback.is_correct === false) return `Falló (${feedback.actual_label})`;
  return feedback.actual_label;
}

export default App;
