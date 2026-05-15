import { useEffect, useRef } from 'react';
import { AreaSeries, ColorType, createChart, type IChartApi } from 'lightweight-charts';

export interface PricePoint {
  timestamp: string;
  open?: number | string;
  high?: number | string;
  low?: number | string;
  close: number | string;
  volume?: number | string;
}

interface ChartProps {
  data: PricePoint[];
  colors?: {
    backgroundColor?: string;
    lineColor?: string;
    textColor?: string;
    areaTopColor?: string;
    areaBottomColor?: string;
  };
}

export function FinancialChart({
  data,
  colors: {
    backgroundColor = '#181b1a',
    lineColor = '#6ee7b7',
    textColor = '#d4d4d8',
    areaTopColor = 'rgba(110, 231, 183, 0.28)',
    areaBottomColor = 'rgba(110, 231, 183, 0.02)',
  } = {},
}: ChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: backgroundColor },
        textColor,
      },
      grid: {
        vertLines: { color: 'rgba(255,255,255,0.06)' },
        horzLines: { color: 'rgba(255,255,255,0.06)' },
      },
      rightPriceScale: {
        borderColor: 'rgba(255,255,255,0.08)',
      },
      timeScale: {
        borderColor: 'rgba(255,255,255,0.08)',
      },
      width: chartContainerRef.current.clientWidth,
      height: 420,
    });

    const series = chart.addSeries(AreaSeries, {
      lineColor,
      topColor: areaTopColor,
      bottomColor: areaBottomColor,
    });

    const formattedData = data
      .map((item) => ({
        time: item.timestamp.split('T')[0],
        value: Number(item.close),
      }))
      .filter((item) => Number.isFinite(item.value))
      .sort((a, b) => a.time.localeCompare(b.time));

    series.setData(formattedData);
    chart.timeScale().fitContent();
    chartRef.current = chart;

    const handleResize = () => {
      if (!chartContainerRef.current) return;
      chartRef.current?.applyOptions({ width: chartContainerRef.current.clientWidth });
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data, backgroundColor, lineColor, textColor, areaTopColor, areaBottomColor]);

  return <div ref={chartContainerRef} className="h-[420px] w-full" />;
}
