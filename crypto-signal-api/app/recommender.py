"""Lógica del Recomendador de Posición, Conversión MXN->USDT y Filtros de Binance (SDD 2.3.3)."""
from __future__ import annotations

import math
import os
import time
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Tuple

import httpx

BINANCE_VISION_BASE = "https://data-api.binance.vision/api/v3"

# Fricción realista idéntica al Programa 1 (SDD 1.3.4: 0.36% round-trip)
ROUND_TRIP_FRICTION_PCT = 0.0036
ENTRY_FEE_PCT = 0.0010

# Filtros base de respaldo por si exchangeInfo no responde a tiempo
DEFAULT_SYMBOL_FILTERS: Dict[str, Dict[str, float]] = {
    "BTCUSDT": {"step_size": 0.00001, "min_qty": 0.00001, "min_notional": 10.0},
    "ETHUSDT": {"step_size": 0.0001, "min_qty": 0.0001, "min_notional": 10.0},
    "SOLUSDT": {"step_size": 0.001, "min_qty": 0.001, "min_notional": 10.0},
    "BNBUSDT": {"step_size": 0.001, "min_qty": 0.001, "min_notional": 10.0},
    "XRPUSDT": {"step_size": 0.1, "min_qty": 0.1, "min_notional": 10.0},
    "ADAUSDT": {"step_size": 0.1, "min_qty": 0.1, "min_notional": 10.0},
    "DOGEUSDT": {"step_size": 1.0, "min_qty": 1.0, "min_notional": 10.0},
    "LINKUSDT": {"step_size": 0.01, "min_qty": 0.01, "min_notional": 10.0},
    "AVAXUSDT": {"step_size": 0.01, "min_qty": 0.01, "min_notional": 10.0},
    "POLUSDT": {"step_size": 0.1, "min_qty": 0.1, "min_notional": 10.0},
    "ATOMUSDT": {"step_size": 0.01, "min_qty": 0.01, "min_notional": 10.0},
    "DOTUSDT": {"step_size": 0.01, "min_qty": 0.01, "min_notional": 10.0},
    "FILUSDT": {"step_size": 0.01, "min_qty": 0.01, "min_notional": 10.0},
    "LTCUSDT": {"step_size": 0.001, "min_qty": 0.001, "min_notional": 10.0},
}

_EXCHANGE_FILTERS_CACHE: Dict[str, Dict[str, float]] = {}
_EXCHANGE_FILTERS_TS: float = 0.0
_USDTMXN_CACHE: Tuple[float, float] = (0.0, 0.0)


async def fetch_usdt_mxn_rate(
    client: httpx.AsyncClient, override_rate: float | None = None
) -> float:
    """Obtiene el tipo de cambio USDT/MXN en vivo desde Binance con caché de 5 min y fallback seguro."""
    if override_rate is not None and override_rate > 0:
        return float(override_rate)

    global _USDTMXN_CACHE
    now = time.time()
    if _USDTMXN_CACHE[0] > 0 and (now - _USDTMXN_CACHE[1]) < 300:
        return _USDTMXN_CACHE[0]

    fallback = float(os.getenv("DEFAULT_USDTMXN_RATE", "20.15"))
    try:
        resp = await client.get(
            f"{BINANCE_VISION_BASE}/ticker/price",
            params={"symbol": "USDTMXN"},
            timeout=1.5,
        )
        if resp.status_code == 200:
            price = float(resp.json().get("price", fallback))
            if price > 5.0:
                _USDTMXN_CACHE = (price, now)
                return price
    except Exception:
        pass

    return fallback


async def get_binance_symbol_filters(
    client: httpx.AsyncClient, symbols: list[str]
) -> Dict[str, Dict[str, float]]:
    """Obtiene y cachea por 1 hora los filtros LOT_SIZE y NOTIONAL de Binance."""
    global _EXCHANGE_FILTERS_CACHE, _EXCHANGE_FILTERS_TS
    now = time.time()
    if _EXCHANGE_FILTERS_CACHE and (now - _EXCHANGE_FILTERS_TS) < 3600:
        return _EXCHANGE_FILTERS_CACHE

    result = {s: DEFAULT_SYMBOL_FILTERS.get(s, {"step_size": 0.01, "min_qty": 0.01, "min_notional": 10.0}).copy() for s in symbols}
    try:
        import json as _json
        symbols_param = _json.dumps(symbols).replace(" ", "")
        resp = await client.get(
            f"{BINANCE_VISION_BASE}/exchangeInfo",
            params={"symbols": symbols_param},
            timeout=2.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            for s_info in data.get("symbols", []):
                sym = s_info.get("symbol")
                if sym not in result:
                    continue
                for f in s_info.get("filters", []):
                    f_type = f.get("filterType")
                    if f_type == "LOT_SIZE":
                        result[sym]["step_size"] = float(f.get("stepSize", result[sym]["step_size"]))
                        result[sym]["min_qty"] = float(f.get("minQty", result[sym]["min_qty"]))
                    elif f_type in ("NOTIONAL", "MIN_NOTIONAL"):
                        result[sym]["min_notional"] = float(
                            f.get("minNotional", result[sym]["min_notional"])
                        )
            _EXCHANGE_FILTERS_CACHE = result
            _EXCHANGE_FILTERS_TS = now
    except Exception:
        pass

    return result


def quantize_lot_size(qty: float, step_size: float) -> float:
    """Redondea hacia abajo (ROUND_DOWN) la cantidad al múltiplo exacto de stepSize de Binance."""
    if qty <= 0 or step_size <= 0:
        return 0.0
    d_qty = Decimal(str(qty))
    d_step = Decimal(str(step_size))
    steps = (d_qty / d_step).to_integral_value(rounding=ROUND_DOWN)
    return float(steps * d_step)


def compute_recommended_position(
    capital_total_usdt: float,
    free_usdt: float,
    risk_pct: float,
    available_slots: int,
    entry_price: float,
    stop_loss: float,
    symbol_filter: Dict[str, float],
) -> Dict[str, float | bool | str]:
    """
    Calcula position_size_usdt y position_qty (SDD 2.3.3):
    - Incluye comisiones + slippage (0.36% RT) en la distancia efectiva al Stop.
    - Respeta el efectivo libre repartido entre los slots disponibles y los filtros de Binance.
    """
    step_size = float(symbol_filter.get("step_size", 0.01))
    min_qty = float(symbol_filter.get("min_qty", 0.01))
    min_notional = max(10.0, float(symbol_filter.get("min_notional", 10.0)))

    if available_slots <= 0 or free_usdt < min_notional or entry_price <= stop_loss or entry_price <= 0:
        return {
            "position_size_usdt": 0.0,
            "position_qty": 0.0,
            "risk_usdt": 0.0,
            "effective_stop_pct": 0.0,
            "min_notional_ok": False,
            "lot_step_size": step_size,
            "note": "Sin slots disponibles o efectivo libre insuficiente.",
        }

    raw_stop_dist_pct = (entry_price - stop_loss) / entry_price
    effective_stop_pct = raw_stop_dist_pct + ROUND_TRIP_FRICTION_PCT

    target_risk_usdt = capital_total_usdt * (risk_pct / 100.0)
    ideal_notional_usdt = target_risk_usdt / effective_stop_pct

    # Límite por liquidez libre disponible (dejando margen para la comisión de entrada)
    max_cash_for_slot = (free_usdt / max(1, available_slots)) / (1.0 + ENTRY_FEE_PCT)
    capped_notional_usdt = min(ideal_notional_usdt, max_cash_for_slot)

    raw_qty = capped_notional_usdt / entry_price
    qty = quantize_lot_size(raw_qty, step_size)
    final_notional_usdt = qty * entry_price
    actual_risk_usdt = final_notional_usdt * effective_stop_pct

    min_notional_ok = (qty >= min_qty) and (final_notional_usdt >= min_notional)
    if not min_notional_ok:
        return {
            "position_size_usdt": round(final_notional_usdt, 2),
            "position_qty": qty,
            "risk_usdt": round(actual_risk_usdt, 2),
            "effective_stop_pct": round(effective_stop_pct * 100.0, 3),
            "min_notional_ok": False,
            "lot_step_size": step_size,
            "note": f"Orden rechazada por filtro Binance: Notional ${final_notional_usdt:.2f} < minNotional ${min_notional:.2f}",
        }

    return {
        "position_size_usdt": round(final_notional_usdt, 2),
        "position_qty": qty,
        "risk_usdt": round(actual_risk_usdt, 2),
        "effective_stop_pct": round(effective_stop_pct * 100.0, 3),
        "min_notional_ok": True,
        "lot_step_size": step_size,
        "note": "Tamaño validado contra filtros LOT_SIZE y minNotional de Binance.",
    }
