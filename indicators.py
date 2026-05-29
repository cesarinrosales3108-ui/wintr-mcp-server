"""
Wintr MCP Server — Technical Indicators
========================================
Calculo de indicadores tecnicos usando TA-Lib y numpy.
Todas las funciones son PURAS (sin estado) para facilitar testing.

Indicadores disponibles:
    - RSI (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    - ATR (Average True Range)
    - Stochastic Oscillator
    - SMA (Simple Moving Average)
    - EMA (Exponential Moving Average)
"""

import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger("wintr-mcp.indicators")

# Intentar importar TA-Lib
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib no instalado. Usando implementacion numpy alternativa.")
    talib = None


def _to_numpy(values) -> np.ndarray:
    """Convierte cualquier secuencia a numpy array."""
    if isinstance(values, np.ndarray):
        return values.astype(float)
    return np.array(values, dtype=float)


def _validate_prices(prices, min_period: int = 2):
    """Valida que los precios sean suficientes para el calculo."""
    prices_np = _to_numpy(prices)
    if len(prices_np) < min_period:
        raise ValueError(f"Se necesitan al menos {min_period} periodos, se recibieron {len(prices_np)}")
    return prices_np


# ── RSI ────────────────────────────────────────────────────

def calculate_rsi(prices, period: int = 14) -> np.ndarray:
    """
    Calcula el Relative Strength Index.
    
    RSI mide la velocidad y magnitud de los cambios de precio.
    > 70: sobrecompra (posible caida)
    < 30: sobreventa (posible subida)
    
    Args:
        prices: Array de precios de cierre
        period: Periodo RSI (default: 14)
    
    Returns:
        Array con valores RSI
    """
    prices_np = _validate_prices(prices, period + 1)
    
    if TALIB_AVAILABLE:
        return talib.RSI(prices_np, timeperiod=period)
    
    # Implementacion numpy alternativa
    deltas = np.diff(prices_np)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros_like(prices_np)
    avg_loss = np.zeros_like(prices_np)
    
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    for i in range(period + 1, len(prices_np)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = 50  # Neutral para periodos iniciales
    
    return rsi


# ── MACD ───────────────────────────────────────────────────

def calculate_macd(prices, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula MACD (Moving Average Convergence Divergence).
    
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal
    
    Args:
        prices: Array de precios de cierre
        fast: Periodo EMA rapida (default: 12)
        slow: Periodo EMA lenta (default: 26)
        signal: Periodo EMA de senal (default: 9)
    
    Returns:
        (macd_line, signal_line, histogram)
    """
    prices_np = _validate_prices(prices, slow)
    
    if TALIB_AVAILABLE:
        macd, signal_line, hist = talib.MACD(prices_np, fast, slow, signal)
        # Rellenar NaN con 0
        macd = np.nan_to_num(macd, nan=0.0)
        signal_line = np.nan_to_num(signal_line, nan=0.0)
        hist = np.nan_to_num(hist, nan=0.0)
        return macd, signal_line, hist
    
    # Implementacion numpy alternativa
    def ema(data, period):
        result = np.zeros_like(data)
        multiplier = 2 / (period + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result
    
    ema_fast = ema(prices_np, fast)
    ema_slow = ema(prices_np, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


# ── Bollinger Bands ────────────────────────────────────────

def calculate_bollinger(prices, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula Bollinger Bands.
    
    Middle = SMA(period)
    Upper = Middle + std_dev * StdDev(prices, period)
    Lower = Middle - std_dev * StdDev(prices, period)
    
    Args:
        prices: Array de precios de cierre
        period: Periodo de la media (default: 20)
        std_dev: Desviaciones estandar (default: 2)
    
    Returns:
        (upper_band, middle_band, lower_band)
    """
    prices_np = _validate_prices(prices, period)
    
    if TALIB_AVAILABLE:
        upper, middle, lower = talib.BBANDS(prices_np, period, std_dev, std_dev)
        upper = np.nan_to_num(upper, nan=0.0)
        middle = np.nan_to_num(middle, nan=0.0)
        lower = np.nan_to_num(lower, nan=0.0)
        return upper, middle, lower
    
    # Implementacion numpy
    sma = calculate_sma(prices_np, period)
    std = np.zeros_like(prices_np)
    
    for i in range(period - 1, len(prices_np)):
        std[i] = np.std(prices_np[i - period + 1:i + 1])
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    upper[:period - 1] = 0
    lower[:period - 1] = 0
    
    return upper, sma, lower


# ── ATR ────────────────────────────────────────────────────

def calculate_atr(high, low, close, period: int = 14) -> np.ndarray:
    """
    Calcula Average True Range (medida de volatilidad).
    
    Args:
        high: Precios maximos
        low: Precios minimos
        close: Precios de cierre
        period: Periodo (default: 14)
    
    Returns:
        Array con valores ATR
    """
    high_np = _to_numpy(high)
    low_np = _to_numpy(low)
    close_np = _to_numpy(close)
    
    if TALIB_AVAILABLE:
        result = talib.ATR(high_np, low_np, close_np, timeperiod=period)
        return np.nan_to_num(result, nan=0.0)
    
    # Implementacion numpy
    tr = np.zeros_like(close_np)
    for i in range(1, len(close_np)):
        hl = high_np[i] - low_np[i]
        hc = abs(high_np[i] - close_np[i - 1])
        lc = abs(low_np[i] - close_np[i - 1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros_like(close_np)
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(close_np)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


# ── Stochastic ─────────────────────────────────────────────

def calculate_stochastic(high, low, close, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula Stochastic Oscillator.
    
    %K = 100 * (Close - Lowest Low) / (Highest High - Lowest Low)
    %D = SMA(%K, d_period)
    
    Args:
        high: Precios maximos
        low: Precios minimos
        close: Precios de cierre
        k_period: Periodo %K (default: 14)
        d_period: Periodo %D (default: 3)
    
    Returns:
        (k_line, d_line)
    """
    if TALIB_AVAILABLE:
        k, d = talib.STOCH(high, low, close, fastk_period=k_period, slowk_period=3,
                           slowk_matype=0, slowd_period=d_period, slowd_matype=0)
        return np.nan_to_num(k, nan=50.0), np.nan_to_num(d, nan=50.0)
    
    # Implementacion numpy
    high_np = _to_numpy(high)
    low_np = _to_numpy(low)
    close_np = _to_numpy(close)
    
    k = np.full_like(close_np, 50.0)
    for i in range(k_period - 1, len(close_np)):
        highest = np.max(high_np[i - k_period + 1:i + 1])
        lowest = np.min(low_np[i - k_period + 1:i + 1])
        if highest != lowest:
            k[i] = 100 * (close_np[i] - lowest) / (highest - lowest)
    
    d = calculate_sma(k, d_period)
    return k, d


# ── Moving Averages ────────────────────────────────────────

def calculate_sma(prices, period: int = 20) -> np.ndarray:
    """
    Calcula Simple Moving Average.
    
    Args:
        prices: Array de precios
        period: Periodo de la media
    
    Returns:
        Array con SMA
    """
    prices_np = _validate_prices(prices, period)
    
    if TALIB_AVAILABLE:
        result = talib.SMA(prices_np, timeperiod=period)
        return np.nan_to_num(result, nan=0.0)
    
    # Implementacion numpy
    result = np.zeros_like(prices_np)
    cumsum = np.cumsum(prices_np, dtype=float)
    result[:period - 1] = 0
    result[period - 1:] = (cumsum[period - 1:] - np.concatenate([[0], cumsum[:len(prices_np) - period]])) / period
    return result


def calculate_ema(prices, period: int = 20) -> np.ndarray:
    """
    Calcula Exponential Moving Average.
    
    Args:
        prices: Array de precios
        period: Periodo de la media
    
    Returns:
        Array con EMA
    """
    prices_np = _validate_prices(prices, period)
    
    if TALIB_AVAILABLE:
        result = talib.EMA(prices_np, timeperiod=period)
        return np.nan_to_num(result, nan=0.0)
    
    # Implementacion numpy
    result = np.zeros_like(prices_np)
    multiplier = 2 / (period + 1)
    result[0] = prices_np[0]
    for i in range(1, len(prices_np)):
        result[i] = (prices_np[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


# ── Interpretacion ─────────────────────────────────────────

def interpret_rsi(rsi_value: float) -> dict:
    """Interpreta el valor actual del RSI."""
    if rsi_value >= 70:
        return {"signal": "SELL", "strength": "strong", "message": "Sobrecompra — posible caida"}
    elif rsi_value >= 60:
        return {"signal": "SELL", "strength": "moderate", "message": "Comprando — cautela"}
    elif rsi_value <= 30:
        return {"signal": "BUY", "strength": "strong", "message": "Sobreventa — posible subida"}
    elif rsi_value <= 40:
        return {"signal": "BUY", "strength": "moderate", "message": "Vendiendo — posible rebote"}
    else:
        return {"signal": "NEUTRAL", "strength": "low", "message": "Rango neutral"}


def interpret_macd(macd_value: float, signal_value: float, histogram: float) -> dict:
    """Interpreta la senal actual del MACD."""
    if macd_value > signal_value and histogram > 0:
        return {"signal": "BUY", "strength": "strong", "message": "Tendencia alcista — MACD por encima de senal"}
    elif macd_value > signal_value:
        return {"signal": "BUY", "strength": "moderate", "message": "Momentum alcista debil"}
    elif macd_value < signal_value and histogram < 0:
        return {"signal": "SELL", "strength": "strong", "message": "Tendencia bajista — MACD por debajo de senal"}
    elif macd_value < signal_value:
        return {"signal": "SELL", "strength": "moderate", "message": "Momentum bajista debil"}
    return {"signal": "NEUTRAL", "strength": "low", "message": "Sin senal clara"}


def interpret_bollinger(price: float, upper: float, middle: float, lower: float) -> dict:
    """Interpreta la posicion del precio en las Bollinger Bands."""
    bb_width = upper - lower
    if bb_width == 0:
        return {"signal": "NEUTRAL", "message": "Bands planas — sin informacion"}
    
    position = (price - middle) / (bb_width / 2)  # -1 a +1
    
    if price >= upper:
        return {"signal": "SELL", "strength": "strong", "message": "Precio tocando banda superior — posible rechazo"}
    elif price <= lower:
        return {"signal": "BUY", "strength": "strong", "message": "Precio tocando banda inferior — posible rebote"}
    elif position > 0.5:
        return {"signal": "SELL", "strength": "moderate", "message": "Precio cerca de banda superior"}
    elif position < -0.5:
        return {"signal": "BUY", "strength": "moderate", "message": "Precio cerca de banda inferior"}
    else:
        return {"signal": "NEUTRAL", "strength": "low", "message": "Precio en rango neutral de las bands"}


def calculate_all_indicators(prices_dict: dict) -> dict:
    """
    Calcula todos los indicadores disponibles y devuelve interpretacion.
    
    Args:
        prices_dict: Dict con 'close', 'high', 'low' arrays
    
    Returns:
        Dict con todos los indicadores e interpretacion
    """
    close = prices_dict.get("close", [])
    high = prices_dict.get("high", close)
    low = prices_dict.get("low", close)
    
    result = {}
    
    # RSI
    try:
        rsi_values = calculate_rsi(close)
        rsi_current = float(rsi_values[-1]) if len(rsi_values) > 0 else 50
        result["rsi"] = {
            "current": round(rsi_current, 2),
            "interpretation": interpret_rsi(rsi_current),
        }
    except Exception as e:
        result["rsi"] = {"error": str(e)}
    
    # MACD
    try:
        macd, signal, hist = calculate_macd(close)
        result["macd"] = {
            "macd_line": round(float(macd[-1]), 5) if len(macd) > 0 else 0,
            "signal_line": round(float(signal[-1]), 5) if len(signal) > 0 else 0,
            "histogram": round(float(hist[-1]), 5) if len(hist) > 0 else 0,
            "interpretation": interpret_macd(
                float(macd[-1]) if len(macd) > 0 else 0,
                float(signal[-1]) if len(signal) > 0 else 0,
                float(hist[-1]) if len(hist) > 0 else 0,
            ),
        }
    except Exception as e:
        result["macd"] = {"error": str(e)}
    
    # Bollinger
    try:
        upper, middle, lower = calculate_bollinger(close)
        current_price = float(close[-1]) if len(close) > 0 else 0
        result["bollinger"] = {
            "upper": round(float(upper[-1]), 5) if len(upper) > 0 else 0,
            "middle": round(float(middle[-1]), 5) if len(middle) > 0 else 0,
            "lower": round(float(lower[-1]), 5) if len(lower) > 0 else 0,
            "width": round(float(upper[-1] - lower[-1]), 5) if len(upper) > 0 and len(lower) > 0 else 0,
            "interpretation": interpret_bollinger(
                current_price,
                float(upper[-1]) if len(upper) > 0 else 0,
                float(middle[-1]) if len(middle) > 0 else 0,
                float(lower[-1]) if len(lower) > 0 else 0,
            ),
        }
    except Exception as e:
        result["bollinger"] = {"error": str(e)}
    
    # ATR
    try:
        import numpy as np
        atr_values = calculate_atr(np.array(high), np.array(low), np.array(close))
        result["atr"] = {
            "current": round(float(atr_values[-1]), 5) if len(atr_values) > 0 else 0,
            "atr_percent": round(float(atr_values[-1] / close[-1] * 100), 2) if len(atr_values) > 0 and len(close) > 0 and close[-1] != 0 else 0,
        }
    except Exception as e:
        result["atr"] = {"error": str(e)}
    
    # Stochastic
    try:
        import numpy as np
        k, d = calculate_stochastic(np.array(high), np.array(low), np.array(close))
        result["stochastic"] = {
            "k": round(float(k[-1]), 2) if len(k) > 0 else 50,
            "d": round(float(d[-1]), 2) if len(d) > 0 else 50,
        }
    except Exception as e:
        result["stochastic"] = {"error": str(e)}
    
    # SMA
    try:
        sma_50 = calculate_sma(close, 50)
        sma_200 = calculate_sma(close, 200)
        current = float(close[-1]) if len(close) > 0 else 0
        sma50_val = float(sma_50[-1]) if len(sma_50) > 0 and not np.isnan(sma_50[-1]) else 0
        sma200_val = float(sma_200[-1]) if len(sma_200) > 0 and not np.isnan(sma_200[-1]) else 0
        
        result["moving_averages"] = {
            "sma_50": round(sma50_val, 5) if sma50_val else None,
            "sma_200": round(sma200_val, 5) if sma200_val else None,
            "price_vs_sma50": round(((current - sma50_val) / sma50_val * 100), 2) if sma50_val else None,
            "price_vs_sma200": round(((current - sma200_val) / sma200_val * 100), 2) if sma200_val else None,
            "golden_cross": sma50_val > sma200_val if sma50_val and sma200_val else None,
        }
    except Exception as e:
        result["moving_averages"] = {"error": str(e)}
    
    return result
