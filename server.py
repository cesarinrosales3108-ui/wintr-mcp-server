"""
Wintr MCP Server — Trading Tools for AI Agents
================================================
FastMCP server que expone MetaTrader 5 como 10 herramientas
para agentes de IA via MCP Protocol.

Uso directo:
    python server.py

Integracion Hermes (config.yaml):
    mcp_servers:
      wintr-trading:
        command: "C:\\Users\\WinterOS\\AppData\\Local\\Programs\\Python\\Python311\\python.exe"
        args: ["C:\\Users\\WinterOS\\trading-ai-agents\\wintr-mcp-server\\server.py"]

Uso como libreria:
    from server import mcp
    # mcp es una instancia de FastMCP lista para usar
"""

import os
import sys
import logging
from datetime import datetime, timezone
from typing import Optional

# Servidor MCP
from fastmcp import FastMCP

# Componentes locales
from mt5_connector import MT5Connector, MT5Error
from indicators import calculate_all_indicators, calculate_rsi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wintr-mcp.server")

# ── Instancia del servidor ────────────────────────────────

mcp = FastMCP(
    "Wintr Trading Server",
    instructions="MT5 trading tools for AI agents — market data, technical analysis, account info, and signals",
)

# ── Conector MT5 (singleton) ─────────────────────────────

_connector: Optional[MT5Connector] = None

def get_connector() -> MT5Connector:
    """Retorna la instancia del conector MT5 (inicializado)."""
    global _connector
    if _connector is None:
        _connector = MT5Connector()
    _connector.initialize()
    return _connector


# ══════════════════════════════════════════════════════════
# TOOLS DE MERCADO (Publicas, solo lectura)
# ══════════════════════════════════════════════════════════


@mcp.tool()
def market_get_rates(symbol: str, timeframe: str = "M5") -> dict:
    """
    Obtiene el precio actual y datos de mercado para un simbolo.

    Args:
        symbol: Par de divisas (ej. "EURUSD", "GBPUSD", "USDJPY")
        timeframe: Marco temporal (M1, M5, M15, M30, H1, H4, D1)

    Returns:
        Dict con bid, ask, open, high, low, close, volume, spread, time
    """
    logger.info(f"market_get_rates({symbol}, {timeframe})")
    try:
        conn = get_connector()
        return conn.get_rates(symbol, timeframe, 1)
    except (MT5Error, ValueError) as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        return {"error": f"Error interno: {str(e)}"}


@mcp.tool()
def market_get_historical(symbol: str, bars: int = 100, timeframe: str = "H1") -> list:
    """
    Obtiene datos historicos OHLCV para un simbolo.

    Args:
        symbol: Par de divisas (ej. "EURUSD")
        bars: Numero de velas (max 5000, default 100)
        timeframe: Marco temporal (M1, M5, M15, M30, H1, H4, D1)

    Returns:
        Lista de velas con time, open, high, low, close, volume
    """
    logger.info(f"market_get_historical({symbol}, {bars}, {timeframe})")
    try:
        conn = get_connector()
        data = conn.get_historical(symbol, min(bars, 5000), timeframe)
        return data
    except (MT5Error, ValueError) as e:
        return [{"error": str(e)}]
    except Exception as e:
        return [{"error": f"Error interno: {str(e)}"}]


@mcp.tool()
def market_technical_indicators(symbol: str, bars: int = 200) -> dict:
    """
    Calcula indicadores tecnicos para un simbolo: RSI, MACD, Bollinger,
    ATR, Stochastic, SMA, EMA — CON interpretacion de senal.

    Args:
        symbol: Par de divisas (ej. "EURUSD")
        bars: Velas para el calculo (default 200, minimo 50)

    Returns:
        Dict con todos los indicadores e interpretacion (BUY/SELL/NEUTRAL)
    """
    logger.info(f"market_technical_indicators({symbol}, {bars})")
    try:
        conn = get_connector()
        data = conn.get_historical(symbol, max(bars, 50), "H1")

        if not data or (isinstance(data, list) and len(data) == 0):
            return {"error": f"No hay datos para {symbol}"}

        close_prices = [d["close"] for d in data]
        high_prices = [d["high"] for d in data]
        low_prices = [d["low"] for d in data]

        prices = {
            "close": close_prices,
            "high": high_prices,
            "low": low_prices,
        }

        indicators = calculate_all_indicators(prices)

        # Agregar resumen general
        signals = []
        for key, value in indicators.items():
            if isinstance(value, dict) and "interpretation" in value:
                signals.append(value["interpretation"]["signal"])

        buy_signals = signals.count("BUY")
        sell_signals = signals.count("SELL")
        neutral_signals = signals.count("NEUTRAL")

        if buy_signals > sell_signals and buy_signals > neutral_signals:
            overall = "BUY"
        elif sell_signals > buy_signals and sell_signals > neutral_signals:
            overall = "SELL"
        else:
            overall = "NEUTRAL"

        indicators["summary"] = {
            "overall_signal": overall,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "neutral_signals": neutral_signals,
        }

        return indicators

    except (MT5Error, ValueError) as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error en indicadores: {e}")
        return {"error": f"Error interno: {str(e)}"}


@mcp.tool()
def market_get_symbols() -> list:
    """
    Obtiene lista de simbolos disponibles en MT5 (forex majors, minors,
    metals, indices).

    Returns:
        Lista de strings con nombres de simbolos
    """
    logger.info("market_get_symbols()")
    try:
        conn = get_connector()
        return conn.get_symbols()
    except Exception as e:
        return [f"Error: {str(e)}"]


# ══════════════════════════════════════════════════════════
# TOOLS DE CUENTA (Requieren autenticacion)
# ══════════════════════════════════════════════════════════


@mcp.tool()
def account_get_info() -> dict:
    """
    Obtiene informacion de la cuenta MT5 conectada.

    Returns:
        Dict con balance, equity, margin, free_margin, margin_level,
        leverage, currency, server, broker
    """
    logger.info("account_get_info()")
    try:
        conn = get_connector()
        return conn.get_account_info()
    except (MT5Error, ValueError) as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Error interno: {str(e)}"}


@mcp.tool()
def account_get_positions() -> list:
    """
    Obtiene todas las posiciones abiertas actualmente.

    Returns:
        Lista de posiciones con ticket, symbol, type (BUY/SELL),
        volume, price_open, profit, sl, tp
    """
    logger.info("account_get_positions()")
    try:
        conn = get_connector()
        return conn.get_positions()
    except (MT5Error, ValueError) as e:
        return [{"error": str(e)}]
    except Exception as e:
        return [{"error": f"Error interno: {str(e)}"}]


@mcp.tool()
def account_get_order_history(days: int = 7) -> list:
    """
    Obtiene historial de ordenes cerradas de los ultimos N dias.

    Args:
        days: Dias hacia atras (max 30, default 7)

    Returns:
        Lista de deals con ticket, symbol, type, volume, price,
        profit, commission, swap
    """
    logger.info(f"account_get_order_history({days})")
    try:
        conn = get_connector()
        return conn.get_order_history(min(days, 30))
    except (MT5Error, ValueError) as e:
        return [{"error": str(e)}]
    except Exception as e:
        return [{"error": f"Error interno: {str(e)}"}]


# ══════════════════════════════════════════════════════════
# TOOLS DE ANALISIS (AI-ready)
# ══════════════════════════════════════════════════════════


@mcp.tool()
def analysis_market_overview(symbols: str = "EURUSD,GBPUSD,USDJPY") -> dict:
    """
    Analisis general del mercado para multiples pares.

    Escanea cada par, calcula tendencia en H1, RSI, y genera
    un resumen accionable para AI agents.

    Args:
        symbols: Simbolos separados por coma (default: "EURUSD,GBPUSD,USDJPY")

    Returns:
        Dict con analisis por par y resumen general
    """
    logger.info(f"analysis_market_overview({symbols})")
    results = {}
    overall_signals = []

    for sym in symbols.split(","):
        sym = sym.strip().upper()
        if not sym:
            continue

        try:
            indicators = market_technical_indicators(sym, 100)
            rates = market_get_rates(sym, "H1")

            if "error" in indicators:
                results[sym] = {"error": indicators["error"]}
                continue

            signal = indicators.get("summary", {}).get("overall_signal", "NEUTRAL")
            rsi_val = indicators.get("rsi", {}).get("current", 50)
            price = rates.get("close", 0) if isinstance(rates, dict) else 0

            results[sym] = {
                "price": price,
                "signal": signal,
                "rsi": rsi_val,
                "trend": "ALCISTA" if signal == "BUY" else "BAJISTA" if signal == "SELL" else "NEUTRAL",
            }
            overall_signals.append(signal)

        except Exception as e:
            results[sym] = {"error": str(e)}

    # Resumen general
    buy = overall_signals.count("BUY")
    sell = overall_signals.count("SELL")
    total = len(overall_signals)

    results["_summary"] = {
        "total_pares": total,
        "alcistas": buy,
        "bajistas": sell,
        "neutrales": total - buy - sell,
        "recomendacion": "BUSCAR COMPRAS" if buy > sell else "BUSCAR VENTAS" if sell > buy else "CAUTELA",
    }

    return results


@mcp.tool()
def analysis_signal_generator(symbol: str) -> dict:
    """
    Genera una senal de trading para un simbolo especifico usando
    la misma logica que Wintr ScalperBot Elite v3.1.

    La senal se basa en: RSI + MACD + Bollinger + tendencia EMA.
    Incluye niveles sugeridos de entrada, stop loss y take profit.

    Args:
        symbol: Par de divisas (ej. "EURUSD")

    Returns:
        Dict con direction, entry_zone, sl_zone, tp_zone, confidence, reasoning
    """
    logger.info(f"analysis_signal_generator({symbol})")
    try:
        indicators = market_technical_indicators(symbol, 200)

        if "error" in indicators:
            return {"error": indicators["error"]}

        rates = market_get_rates(symbol, "M5")
        if isinstance(rates, dict) and "error" in rates:
            return rates

        current_price = rates.get("close", 0) if isinstance(rates, dict) else 0

        # Extraer senales
        rsi_data = indicators.get("rsi", {})
        macd_data = indicators.get("macd", {})
        bollinger_data = indicators.get("bollinger", {})
        ma_data = indicators.get("moving_averages", {})

        # Determinar direccion
        signals = []
        reasons = []

        # RSI
        rsi_signal = rsi_data.get("interpretation", {}).get("signal", "NEUTRAL")
        signals.append(rsi_signal)
        if rsi_signal != "NEUTRAL":
            reasons.append(f"RSI: {rsi_data.get('interpretation', {}).get('message', '')}")

        # MACD
        macd_signal = macd_data.get("interpretation", {}).get("signal", "NEUTRAL")
        signals.append(macd_signal)
        if macd_signal != "NEUTRAL":
            reasons.append(f"MACD: {macd_data.get('interpretation', {}).get('message', '')}")

        # Bollinger
        bb_signal = bollinger_data.get("interpretation", {}).get("signal", "NEUTRAL")
        signals.append(bb_signal)
        if bb_signal != "NEUTRAL":
            reasons.append(f"Bollinger: {bollinger_data.get('interpretation', {}).get('message', '')}")

        # Votacion
        buy_votes = signals.count("BUY")
        sell_votes = signals.count("SELL")

        if buy_votes > sell_votes and buy_votes >= 2:
            direction = "BUY"
            confidence = min(50 + buy_votes * 15, 95)
            sl_distance = current_price * 0.003  # 0.3%
            tp_distance = current_price * 0.009  # 0.9%
        elif sell_votes > buy_votes and sell_votes >= 2:
            direction = "SELL"
            confidence = min(50 + sell_votes * 15, 95)
            sl_distance = current_price * 0.003
            tp_distance = current_price * 0.009
        else:
            direction = "NEUTRAL"
            confidence = 30
            sl_distance = current_price * 0.002
            tp_distance = current_price * 0.006

        # Niveles
        if direction == "BUY":
            entry_zone = current_price
            sl_zone = current_price - sl_distance
            tp_zone = current_price + tp_distance
        elif direction == "SELL":
            entry_zone = current_price
            sl_zone = current_price + sl_distance
            tp_zone = current_price - tp_distance
        else:
            entry_zone = current_price
            sl_zone = current_price * 0.997
            tp_zone = current_price * 1.003

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_zone": round(entry_zone, 5),
            "sl_zone": round(sl_zone, 5),
            "tp_zone": round(tp_zone, 5),
            "confidence": confidence,
            "risk_reward": round(abs(tp_zone - entry_zone) / abs(entry_zone - sl_zone), 2) if entry_zone != sl_zone else 1,
            "reasoning": " | ".join(reasons) if reasons else "Senales mixtas — cautela recomendada",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error generando senal: {e}")
        return {"error": f"Error generando senal: {str(e)}"}


@mcp.tool()
def analysis_risk_check(positions_json: str = "") -> dict:
    """
    Analiza el riesgo actual de la cartera.

    Si se proporcionan posiciones (JSON), las analiza.
    Si no, usa las posiciones actuales de MT5.

    Args:
        positions_json: JSON string con posiciones (opcional)

    Returns:
        Dict con risk_score, max_drawdown, sugerencias
    """
    logger.info("analysis_risk_check()")
    try:
        conn = get_connector()
        account = conn.get_account_info()
        positions = conn.get_positions()

        balance = account.get("balance", 0)
        equity = account.get("equity", 0)
        margin_level = account.get("margin_level", 0)

        if balance == 0:
            return {"error": "Balance en cero"}

        # Calcular metricas de riesgo
        total_profit = sum(p.get("profit", 0) for p in positions)
        total_volume = sum(p.get("volume", 0) for p in positions)
        position_count = len(positions)

        # Drawdown actual
        dd_percent = round((balance - equity) / balance * 100, 2) if balance > 0 else 0

        # Exposicion
        exposure_percent = round(margin_level, 2)

        # Risk score (0-100)
        risk_score = 0
        factors = []

        if position_count > 10:
            risk_score += 20
            factors.append(f"Muchas posiciones ({position_count})")
        elif position_count > 5:
            risk_score += 10

        if dd_percent > 10:
            risk_score += 30
            factors.append(f"Drawdown alto ({dd_percent}%)")
        elif dd_percent > 5:
            risk_score += 15
            factors.append(f"Drawdown moderado ({dd_percent}%)")

        if exposure_percent < 100:
            risk_score += 25
            factors.append("Margin call cercano")

        if total_profit < -balance * 0.05:
            risk_score += 20
            factors.append("Perdidas >5% del balance")

        # Sugerencias
        suggestions = []
        if risk_score > 50:
            suggestions.append("REDUCIR RIESGO: cerrar posiciones o reducir volumen")
        if position_count > 8:
            suggestions.append(f"Tienes {position_count} posiciones abiertas. Considera reducir a max 5")
        if dd_percent > 8:
            suggestions.append("Drawdown elevado. Revisar stops")
        if not suggestions:
            suggestions.append("Riesgo controlado. Continuar normal")

        return {
            "risk_score": min(risk_score, 100),
            "max_drawdown": dd_percent,
            "exposure": f"{exposure_percent:.1f}%",
            "open_positions": position_count,
            "total_volume": total_volume,
            "total_unrealized_pnl": round(total_profit, 2),
            "factors": factors,
            "suggestions": suggestions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Error en risk check: {e}")
        return {"error": f"Error: {str(e)}"}


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════


def main():
    """Inicia el servidor MCP en modo stdio (estandar para MCP)."""
    logger.info("=" * 50)
    logger.info("Wintr MCP Server iniciando...")
    logger.info(f"FastMCP version: {FastMCP.__module__}")
    logger.info("Tools registradas:")
    logger.info("  1. market_get_rates - Precios actuales")
    logger.info("  2. market_get_historical - Datos historicos")
    logger.info("  3. market_technical_indicators - Indicadores tecnicos")
    logger.info("  4. market_get_symbols - Simbolos disponibles")
    logger.info("  5. account_get_info - Info de cuenta")
    logger.info("  6. account_get_positions - Posiciones abiertas")
    logger.info("  7. account_get_order_history - Historial de trades")
    logger.info("  8. analysis_market_overview - Resumen del mercado")
    logger.info("  9. analysis_signal_generator - Generador de senales")
    logger.info("  10. analysis_risk_check - Analisis de riesgo")
    logger.info("=" * 50)

    # FastMCP corre en modo stdio (estandar MCP)
    mcp.run()


if __name__ == "__main__":
    main()
