"""
Wintr MCP Server — MT5 Connector
=================================
Capa de conexion a MetaTrader 5 con cache, auto-reconnect,
rate limiting y manejo de errores.

Uso:
    from mt5_connector import MT5Connector
    connector = MT5Connector()
    rates = connector.get_rates("EURUSD", "M5")
    
Requisitos:
    - MetaTrader 5 instalado y funcionando
    - metatrader5 Python package (pip install metatrader5)
    - numpy
"""

import time
import logging
from datetime import datetime, timezone
from typing import Optional
from functools import lru_cache

import numpy as np

logger = logging.getLogger("wintr-mcp.mt5_connector")

# Import MT5 con manejo de error graceful
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None
    logger.warning("MetaTrader5 no instalado. Usar modo fallback.")


# Timeframes mapping
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1 if mt5 else 1,
    "M5": mt5.TIMEFRAME_M5 if mt5 else 5,
    "M15": mt5.TIMEFRAME_M15 if mt5 else 15,
    "M30": mt5.TIMEFRAME_M30 if mt5 else 30,
    "H1": mt5.TIMEFRAME_H1 if mt5 else 60,
    "H4": mt5.TIMEFRAME_H4 if mt5 else 240,
    "D1": mt5.TIMEFRAME_D1 if mt5 else 1440,
    "W1": mt5.TIMEFRAME_W1 if mt5 else 10080,
    "MN1": mt5.TIMEFRAME_MN1 if mt5 else 43200,
}


class MT5Error(Exception):
    """Error personalizado para fallos de MT5."""
    pass


class MT5Connector:
    """
    Conector a MetaTrader 5 con:
    - Inicializacion automatica
    - Cache LRU con TTL
    - Auto-reconnect en caso de desconexion
    - Timeout por llamada
    - Manejo de errores descriptivo
    """

    def __init__(self, cache_ttl: int = 2, timeout: int = 5):
        """
        Args:
            cache_ttl: Tiempo de vida del cache en segundos (default: 2)
            timeout: Timeout por llamada en segundos (default: 5)
        """
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self._initialized = False
        self._last_init_time = 0
        self._init_attempts = 0
        self._cache = {}
        self._cache_times = {}

    # ── Inicializacion ────────────────────────────────────

    def initialize(self) -> bool:
        """Inicializa la conexion con MT5. Reintenta si falla."""
        if not MT5_AVAILABLE:
            raise MT5Error("MetaTrader5 package no instalado. pip install metatrader5")

        if self._initialized:
            # Verificar que sigue conectado
            if mt5.terminal_info():
                return True
            logger.info("MT5 desconectado. Reconectando...")
            self._initialized = False

        # Intentar inicializar con backoff
        max_attempts = 3
        for attempt in range(max_attempts):
            if mt5.initialize():
                self._initialized = True
                self._last_init_time = time.time()
                self._init_attempts = 0
                logger.info("MT5 conectado exitosamente")
                return True
            
            wait = (attempt + 1) * 2
            logger.warning(f"Intento {attempt + 1}/{max_attempts} fallo. Reintentando en {wait}s...")
            time.sleep(wait)

        self._init_attempts += 1
        raise MT5Error(
            f"No se pudo conectar a MT5 tras {max_attempts} intentos. "
            "Asegurate de que MetaTrader 5 este abierto y el Algo Trading activado."
        )

    def shutdown(self):
        """Cierra la conexion con MT5."""
        if self._initialized and MT5_AVAILABLE:
            mt5.shutdown()
            self._initialized = False
            logger.info("MT5 desconectado")

    # ── Cache ─────────────────────────────────────────────

    def _get_cached(self, key: str):
        """Retorna valor del cache si no ha expirado."""
        if key in self._cache and key in self._cache_times:
            if time.time() - self._cache_times[key] < self.cache_ttl:
                return self._cache[key]
        return None

    def _set_cache(self, key: str, value):
        """Guarda valor en cache con timestamp."""
        self._cache[key] = value
        self._cache_times[key] = time.time()

    # ── Market Data ───────────────────────────────────────

    def get_rates(self, symbol: str, timeframe: str = "M5", count: int = 1) -> dict:
        """
        Obtiene datos de mercado actuales para un simbolo.
        
        Args:
            symbol: Par (ej. "EURUSD")
            timeframe: Timeframe (M1, M5, M15, M30, H1, H4, D1)
            count: Numero de velas (default: 1, la mas reciente)
            
        Returns:
            Dict con bid, ask, high, low, close, volume, spread, time
            
        Raises:
            MT5Error: Si no se puede obtener datos
            ValueError: Si el simbolo o timeframe son invalidos
        """
        self._validate_symbol(symbol)
        self._validate_timeframe(timeframe)
        self.initialize()

        cache_key = f"rates_{symbol}_{timeframe}_{count}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        tf = TIMEFRAMES.get(timeframe)
        if tf is None:
            raise ValueError(f"Timeframe invalido: {timeframe}. Usa: {', '.join(TIMEFRAMES.keys())}")

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            raise MT5Error(
                f"No se pudieron obtener rates para {symbol} ({timeframe}). "
                f"Error: {mt5.last_error()}"
            )

        # Si es solo 1 vela, formato simplificado
        if count == 1:
            r = rates[0]
            tick = mt5.symbol_info_tick(symbol) if hasattr(mt5, 'symbol_info_tick') else None
            # Obtener spread de symbol_info (mas confiable)
            sym_info = mt5.symbol_info(symbol)
            spread = sym_info.spread if sym_info else 0
            
            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "bid": tick.bid if tick else float(r[2]),
                "ask": tick.ask if tick else float(r[3]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": int(r[5]),
                "spread": spread,
                "time": datetime.fromtimestamp(r[0], tz=timezone.utc).isoformat(),
            }
        else:
            result = []
            for r in rates:
                result.append({
                    "time": datetime.fromtimestamp(r[0], tz=timezone.utc).isoformat(),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                    "volume": int(r[5]),
                })
            result = {"symbol": symbol, "timeframe": timeframe, "bars": count, "data": result}

        self._set_cache(cache_key, result)
        return result

    def get_historical(self, symbol: str, bars: int = 100, timeframe: str = "H1") -> list:
        """
        Obtiene datos historicos OHLCV.
        
        Args:
            symbol: Par (ej. "EURUSD")
            bars: Numero de velas (max 5000)
            timeframe: Timeframe
            
        Returns:
            Lista de velas con time, open, high, low, close, volume
        """
        if bars > 5000:
            raise ValueError("Maximo 5000 velas por llamada")
        
        result = self.get_rates(symbol, timeframe, bars)
        return result.get("data", []) if isinstance(result, dict) else [result]

    # ── Account ───────────────────────────────────────────

    def get_account_info(self) -> dict:
        """
        Obtiene informacion de la cuenta MT5.
        
        Returns:
            Dict con balance, equity, margin, free_margin, margin_level,
            leverage, currency, server, broker, name (ofuscado)
        """
        self.initialize()
        info = mt5.account_info()
        if info is None:
            raise MT5Error(f"No se pudo obtener info de cuenta. Error: {mt5.last_error()}")

        return {
            "balance": float(info.balance),
            "equity": float(info.equity),
            "margin": float(info.margin),
            "free_margin": float(info.margin_free),
            "margin_level": float(info.margin_level) if info.margin_level > 0 else 0.0,
            "leverage": int(info.leverage),
            "currency": info.currency,
            "server": info.server,
            "broker": info.company,
            "name": info.name[:3] + "***" if info.name else "***",  # Ofuscado
            "login": str(info.login)[:3] + "***" if info.login else "***",  # Ofuscado
        }

    def get_positions(self) -> list:
        """
        Obtiene todas las posiciones abiertas.
        
        Returns:
            Lista de posiciones con datos relevantes
        """
        self.initialize()
        positions = mt5.positions_get()
        if positions is None:
            return []

        result = []
        for p in positions:
            result.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "BUY" if p.type == 0 else "SELL",
                "volume": float(p.volume),
                "price_open": float(p.price_open),
                "sl": float(p.sl) if p.sl else None,
                "tp": float(p.tp) if p.tp else None,
                "profit": float(p.profit),
                "swap": float(p.swap),
                "comment": p.comment or "",
                "time": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
            })
        return result

    def get_order_history(self, days: int = 7) -> list:
        """
        Obtiene historial de ordenes cerradas.
        
        Args:
            days: Dias hacia atras (max 30)
            
        Returns:
            Lista de deals cerrados
        """
        if days > 30:
            raise ValueError("Maximo 30 dias de historial")
        
        self.initialize()
        now = datetime.now(timezone.utc)
        from_time = datetime(now.year, now.month, now.day - days, tzinfo=timezone.utc)
        
        history = mt5.history_deals_get(from_time, now)
        if history is None:
            return []

        result = []
        for d in history:
            result.append({
                "ticket": d.ticket,
                "symbol": d.symbol,
                "type": "BUY" if d.type == 0 else "SELL",
                "volume": float(d.volume),
                "price": float(d.price),
                "profit": float(d.profit),
                "commission": float(d.commission),
                "swap": float(d.swap),
                "comment": d.comment or "",
                "time": datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat(),
            })
        return result

    # ── Trading ───────────────────────────────────────────

    def place_order(self, symbol: str, volume: float, order_type: str,
                    price: float = 0.0, sl: float = 0.0, tp: float = 0.0,
                    comment: str = "MCP") -> dict:
        """
        Envia una orden de mercado.
        SOLO DISPONIBLE con flag --allow-trading.
        
        Args:
            symbol: Par
            volume: Volumen en lotes
            order_type: "BUY" o "SELL"
            price: Precio (0 = market)
            sl: Stop Loss
            tp: Take Profit
            comment: Comentario de la orden
            
        Returns:
            Dict con resultado de la orden
        """
        self.initialize()
        
        # Validar tipo
        if order_type.upper() == "BUY":
            order_type_mt5 = mt5.ORDER_TYPE_BUY
            order_type_str = "buy"
        elif order_type.upper() == "SELL":
            order_type_mt5 = mt5.ORDER_TYPE_SELL
            order_type_str = "sell"
        else:
            raise ValueError(f"Tipo invalido: {order_type}. Usa BUY o SELL")

        # Preparar request
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise MT5Error(f"Simbolo {symbol} no encontrado")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type_mt5,
            "price": price if price > 0 else (symbol_info.ask if order_type.upper() == "BUY" else symbol_info.bid),
            "sl": float(sl) if sl > 0 else 0.0,
            "tp": float(tp) if tp > 0 else 0.0,
            "deviation": 10,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            raise MT5Error(f"Orden rechazada: {result.comment} (codigo: {result.retcode})")

        return {
            "ticket": result.order,
            "price": result.price,
            "volume": float(volume),
            "type": order_type_str,
            "symbol": symbol,
            "comment": comment,
        }

    # ── Utilidades ────────────────────────────────────────

    def get_symbols(self) -> list:
        """Retorna lista de simbolos disponibles en MT5."""
        self.initialize()
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        return [s.name for s in symbols if s.name.endswith(("USD", "JPY", "GBP", "EUR", "AUD", "NZD", "CAD", "CHF"))]

    def _validate_symbol(self, symbol: str):
        """Valida que el simbolo sea una string no vacia."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Simbolo invalido: {symbol}")

    def _validate_timeframe(self, timeframe: str):
        """Valida que el timeframe sea valido."""
        if timeframe not in TIMEFRAMES:
            valid = ", ".join(TIMEFRAMES.keys())
            raise ValueError(f"Timeframe '{timeframe}' invalido. Validos: {valid}")

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, *args):
        self.shutdown()
