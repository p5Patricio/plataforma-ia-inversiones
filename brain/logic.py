import pandas as pd
import numpy as np

def calculate_sma(df, period=20):
    """Calcula la Media Móvil Simple."""
    return df['close'].rolling(window=period).mean()

def calculate_ema(df, period=20):
    """Calcula la Media Móvil Exponencial."""
    return df['close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(df, period=14):
    """Calcula el Relative Strength Index (RSI)."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calcula el MACD y su línea de señal."""
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def generate_signals(prices_json):
    """
    Toma una lista de precios en formato JSON, calcula indicadores
    y devuelve una señal de inversión basada en reglas lógicas.
    """
    if not prices_json or len(prices_json) < 30:
        return {"signal": "NEUTRAL", "confidence": 0, "reason": "Datos insuficientes"}

    # Convertir a DataFrame y ordenar cronológicamente
    df = pd.DataFrame(prices_json)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # Calcular indicadores
    df['rsi'] = calculate_rsi(df)
    df['sma_20'] = calculate_sma(df, 20)
    df['ema_10'] = calculate_ema(df, 10)
    macd, macd_signal = calculate_macd(df)
    df['macd'] = macd
    df['macd_signal'] = macd_signal

    # Últimos valores para la decisión
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    score = 0
    reasons = []

    # Lógica de decisión (Simulando un modelo de ensamble)
    # 1. RSI
    if last['rsi'] < 30:
        score += 0.4
        reasons.append("Sobreventa (RSI bajo)")
    elif last['rsi'] > 70:
        score -= 0.4
        reasons.append("Sobrecompra (RSI alto)")

    # 2. Cruce de Medias / Tendencia
    if last['close'] > last['sma_20']:
        score += 0.3
        reasons.append("Precio por encima de media móvil 20")
    else:
        score -= 0.3
        reasons.append("Precio por debajo de media móvil 20")

    # 3. MACD Momentum
    if last['macd'] > last['macd_signal'] and prev['macd'] <= prev['macd_signal']:
        score += 0.3
        reasons.append("Cruce alcista de MACD")
    elif last['macd'] < last['macd_signal'] and prev['macd'] >= prev['macd_signal']:
        score -= 0.3
        reasons.append("Cruce bajista de MACD")

    # Determinar señal final
    if score >= 0.5:
        signal = "BUY"
    elif score <= -0.5:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "confidence": round(min(abs(score), 1.0), 2),
        "indicators": {
            "rsi": round(last['rsi'], 2),
            "close": last['close'],
            "sma_20": round(last['sma_20'], 2)
        },
        "reasons": reasons
    }
