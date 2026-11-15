"""Escáner paralelo asíncrono (<2.5s) con Wilder RMA exacto y filtro de noticias/F&G (SDD 2.3)."""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

import httpx
import numpy as np
import pandas as pd

from .models import (
    ScanRequest,
    ScanResponse,
    SignalCandidate,
    StrategyManifest,
    StrategyParameters,
)
from .recommender import (
    BINANCE_VISION_BASE,
    compute_recommended_position,
    fetch_usdt_mxn_rate,
    get_binance_symbol_filters,
)

MAX_OPEN_POSITIONS = 3


def wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Suavizado de Wilder (RMA) idéntico a TradingView ta.rma()."""
    values = series.to_numpy(dtype=float, copy=True)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return pd.Series(out, index=series.index)

    valid_idx = np.where(~np.isnan(values))[0]
    if len(valid_idx) < period:
        return pd.Series(out, index=series.index)

    first_valid = valid_idx[0]
    seed_end = first_valid + period
    if seed_end > n:
        return pd.Series(out, index=series.index)

    out[seed_end - 1] = np.nanmean(values[first_valid:seed_end])
    alpha = 1.0 / float(period)
    one_minus_alpha = 1.0 - alpha

    for i in range(seed_end, n):
        val = values[i]
        out[i] = out[i - 1] if np.isnan(val) else (alpha * val + one_minus_alpha * out[i - 1])

    return pd.Series(out, index=series.index)


def compute_rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = wilder_rma(gain, period)
    avg_loss = wilder_rma(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.where(avg_loss == 0.0, 100.0, rsi)
    rsi = np.where((avg_gain == 0.0) & (avg_loss == 0.0), 50.0, rsi)
    return pd.Series(rsi, index=close.index)


def compute_atr_wilder(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    if len(true_range) > 0:
        true_range.iloc[0] = high.iloc[0] - low.iloc[0]
    return wilder_rma(true_range, period)


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    values = series.to_numpy(dtype=float, copy=True)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n < span:
        return pd.Series(out, index=series.index)
    out[span - 1] = np.nanmean(values[:span])
    alpha = 2.0 / (span + 1.0)
    for i in range(span, n):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)


async def _fetch_klines(
    client: httpx.AsyncClient, symbol: str, interval: str, limit: int
) -> pd.DataFrame:
    resp = await client.get(
        f"{BINANCE_VISION_BASE}/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
        timeout=2.2,
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "qav",
            "trades",
            "tbb",
            "tbq",
            "ig",
        ],
    )[["open_time", "open", "high", "low", "close", "volume", "close_time"]]

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)

    # Descartar la última vela si aún está en formación (evita repintado y look-ahead)
    now_utc = pd.Timestamp(datetime.now(timezone.utc))
    # Si el reloj local del entorno es posterior al de Binance, igual descartamos la última fila abierta si quedan >200
    if len(df) > 2:
        last_close = df["close_time"].iloc[-1]
        if last_close > now_utc:
            df = df.iloc[:-1].reset_index(drop=True)
        else:
            # En vivo en Binance, la última vela devuelta por /klines sin endTime siempre es la vela en curso
            df = df.iloc[:-1].reset_index(drop=True)

    return df


async def _fetch_fear_and_greed(client: httpx.AsyncClient) -> Tuple[int, str]:
    try:
        resp = await client.get("https://api.alternative.me/fng/?limit=1", timeout=1.2)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                val = int(data[0].get("value", 50))
                lbl = str(data[0].get("value_classification", "Neutral"))
                return val, lbl
    except Exception:
        pass
    return 50, "Neutral"


async def _fetch_cryptopanic_bearish_flags(
    client: httpx.AsyncClient, symbols: List[str]
) -> Dict[str, str]:
    """Consulta CryptoPanic si existe API Key configurada; devuelve sentimiento por moneda."""
    api_key = os.getenv("CRYPTOPANIC_API_KEY", "").strip()
    out = {s: "NEUTRAL" for s in symbols}
    if not api_key:
        return out

    currencies = ",".join(s.replace("USDT", "") for s in symbols)
    try:
        resp = await client.get(
            "https://cryptopanic.com/api/v1/posts/",
            params={
                "auth_token": api_key,
                "currencies": currencies,
                "filter": "important",
                "public": "true",
            },
            timeout=1.2,
        )
        if resp.status_code == 200:
            posts = resp.json().get("results", [])
            for post in posts:
                votes = post.get("votes", {})
                neg = int(votes.get("negative", 0)) + int(votes.get("toxic", 0))
                pos = int(votes.get("positive", 0))
                for c in post.get("currencies", []) or []:
                    code = f"{c.get('code', '')}USDT"
                    if code in out:
                        if neg > pos + 3:
                            out[code] = "BEARISH_ALERT"
                        elif pos > neg + 3 and out[code] != "BEARISH_ALERT":
                            out[code] = "BULLISH"
    except Exception:
        pass
    return out


def _evaluate_symbol_snapshot(
    symbol: str,
    df_1d: pd.DataFrame,
    df_4h: pd.DataFrame,
    params: StrategyParameters,
) -> Dict | None:
    if len(df_1d) < params.ema_macro_1d + 2 or len(df_4h) < params.ema_slow_4h + 5:
        return None

    ema200_1d_series = compute_ema(df_1d["close"], params.ema_macro_1d)
    ema200_1d_yesterday = float(ema200_1d_series.iloc[-1])

    h4 = df_4h.copy()
    h4["ema_fast_4h"] = compute_ema(h4["close"], params.ema_fast_4h)
    h4["ema_slow_4h"] = compute_ema(h4["close"], params.ema_slow_4h)
    h4["rsi_4h"] = compute_rsi_wilder(h4["close"], params.rsi_period)
    h4["atr_4h"] = compute_atr_wilder(h4["high"], h4["low"], h4["close"], params.atr_period)
    h4["atr_pct"] = h4["atr_4h"] / h4["close"]

    pb_window = max(1, int(params.pullback_lookback_bars))
    brk_window = max(2, int(params.breakout_lookback_bars))

    h4["recent_min_low"] = h4["low"].shift(1).rolling(pb_window).min()
    h4["pullback_upper"] = (
        h4["ema_fast_4h"].shift(1) + params.pullback_atr_tolerance * h4["atr_4h"].shift(1)
    )
    h4["structure_high"] = h4["high"].shift(1).rolling(brk_window).max()
    h4["vol_sma20"] = h4["volume"].shift(1).rolling(20).mean()

    last = h4.iloc[-1]
    prev3_ema_fast = float(h4["ema_fast_4h"].iloc[-4])

    close_val = float(last["close"])
    ema_fast_val = float(last["ema_fast_4h"])
    ema_slow_val = float(last["ema_slow_4h"])
    rsi_val = float(last["rsi_4h"])
    atr_val = float(last["atr_4h"])
    atr_pct_val = float(last["atr_pct"])

    if any(np.isnan(x) for x in [close_val, ema_fast_val, ema_slow_val, rsi_val, atr_val, ema200_1d_yesterday]):
        return None

    cond_macro = close_val > ema200_1d_yesterday
    cond_trend_4h = (ema_fast_val > ema_slow_val) and (ema_fast_val > prev3_ema_fast)
    cond_volatility = (atr_pct_val >= params.min_atr_pct) and (atr_pct_val <= params.max_atr_pct)
    cond_pullback = float(last["recent_min_low"]) <= float(last["pullback_upper"])
    cond_breakout = (close_val > float(last["structure_high"])) and (
        float(last["volume"]) >= params.volume_mult_min * float(last["vol_sma20"])
    )
    cond_rsi = (rsi_val >= params.rsi_entry_min) and (rsi_val <= params.rsi_entry_max)

    is_signal_ready = bool(
        cond_macro
        and cond_trend_4h
        and cond_volatility
        and cond_pullback
        and cond_breakout
        and cond_rsi
    )
    is_uptrend_watchlist = bool(cond_macro and (ema_fast_val > ema_slow_val))

    roc_21 = (close_val / float(h4["close"].iloc[-22])) - 1.0 if len(h4) >= 22 else 0.0
    rs_score = float(roc_21 / atr_pct_val) if atr_pct_val > 0 else -999.0

    notes: List[str] = []
    if is_signal_ready:
        notes.append("Gatillo 4H confirmado: Pullback a EMA21 + Ruptura de estructura con volumen.")
    else:
        if not cond_macro:
            notes.append("Fuera de régimen macro (Close 4H <= EMA200 1D de ayer).")
        if not cond_trend_4h:
            notes.append("Estructura 4H sin alineación alcista EMA21 > EMA50.")
        if cond_macro and cond_trend_4h and not cond_breakout:
            notes.append(
                f"En tendencia alcista; esperando ruptura sobre {float(last['structure_high']):.4f} con volumen."
            )

    candle_close_dt = pd.Timestamp(last["open_time"]) + pd.Timedelta(hours=4)
    valid_until_dt = candle_close_dt + pd.Timedelta(hours=4)

    return {
        "symbol": symbol,
        "is_signal_ready": is_signal_ready,
        "is_watchlist": is_uptrend_watchlist,
        "candle_close_utc": candle_close_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_until_utc": valid_until_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "entry_price": close_val,
        "atr_4h": atr_val,
        "atr_pct": atr_pct_val,
        "rsi_4h": rsi_val,
        "ema_fast_4h": ema_fast_val,
        "ema_slow_4h": ema_slow_val,
        "ema_macro_1d": ema200_1d_yesterday,
        "rs_score": rs_score,
        "notes": notes,
    }


async def execute_market_scan(
    manifest: StrategyManifest,
    req: ScanRequest,
) -> ScanResponse:
    """Ejecuta el escaneo paralelo de los 10 activos aprobados en <2.5 segundos."""
    t0 = time.perf_counter()
    symbols = manifest.approved_universe[:10]
    params = manifest.parameters

    open_set = {s.upper().strip() for s in req.open_positions if s.strip()}
    available_slots = max(0, MAX_OPEN_POSITIONS - len(open_set))

    async with httpx.AsyncClient(
        headers={"User-Agent": "crypto-signal-api/5.0"},
        limits=httpx.Limits(max_connections=25, max_keepalive_connections=15),
    ) as client:
        # Lanzamos todas las corrutinas en paralelo con asyncio.gather (SDD 2.3.1)
        tasks_1d = [_fetch_klines(client, s, "1d", 240) for s in symbols]
        tasks_4h = [_fetch_klines(client, s, "4h", 90) for s in symbols]

        all_results = await asyncio.gather(
            fetch_usdt_mxn_rate(client, req.usdt_mxn_override),
            get_binance_symbol_filters(client, symbols),
            _fetch_fear_and_greed(client),
            _fetch_cryptopanic_bearish_flags(client, symbols),
            *tasks_1d,
            *tasks_4h,
            return_exceptions=True,
        )

    usdt_mxn_rate = float(all_results[0]) if not isinstance(all_results[0], Exception) else 20.15
    symbol_filters = (
        all_results[1] if not isinstance(all_results[1], Exception) else {}
    )
    fg_tuple = all_results[2] if not isinstance(all_results[2], Exception) else (50, "Neutral")
    news_map = all_results[3] if not isinstance(all_results[3], Exception) else {}

    fg_val, fg_label = fg_tuple
    n_sym = len(symbols)
    klines_1d = all_results[4 : 4 + n_sym]
    klines_4h = all_results[4 + n_sym : 4 + 2 * n_sym]

    capital_total_usdt = round(req.capital_total_mxn / usdt_mxn_rate, 2)

    # Circuit Breaker por pánico sistémico extremo o estrategia no aprobada
    circuit_breaker_active = False
    circuit_breaker_reason = None
    if not manifest.strategy_approved:
        circuit_breaker_active = True
        circuit_breaker_reason = "Estrategia bloqueada en strategy_manifest.json (strategy_approved=false)."
    elif fg_val <= 10:
        circuit_breaker_active = True
        circuit_breaker_reason = f"Circuit Breaker activo por Pánico Extremo (Fear & Greed = {fg_val})."

    raw_snapshots: List[Dict] = []
    for idx, sym in enumerate(symbols):
        if sym in open_set:
            continue
        df1 = klines_1d[idx]
        df4 = klines_4h[idx]
        if isinstance(df1, Exception) or isinstance(df4, Exception):
            continue
        snap = _evaluate_symbol_snapshot(sym, df1, df4, params)
        if snap is None:
            continue
        if snap["is_signal_ready"] or (req.include_watchlist and snap["is_watchlist"]):
            raw_snapshots.append(snap)

    # Ordenar primero SIGNAL_READY y luego por Fuerza Relativa (rs_score) descendente
    raw_snapshots.sort(
        key=lambda x: (1 if x["is_signal_ready"] else 0, x["rs_score"]),
        reverse=True,
    )

    candidates: List[SignalCandidate] = []
    slots_left_for_sizing = available_slots
    free_usdt_remaining = req.free_usdt

    for snap in raw_snapshots:
        sym = snap["symbol"]
        entry_price = float(snap["entry_price"])
        atr_val = float(snap["atr_4h"])
        stop_dist = params.sl_atr_mult * atr_val
        stop_loss = max(1e-8, entry_price - stop_dist)
        take_profit = entry_price + (params.tp_r_mult * stop_dist)
        max_chase_price = entry_price + (0.35 * atr_val)

        sentiment = news_map.get(sym, "NEUTRAL")
        notes = list(snap["notes"])

        # Si hay alerta bajista severa en CryptoPanic, degradar SIGNAL_READY a WATCHLIST
        is_ready = bool(snap["is_signal_ready"]) and not circuit_breaker_active
        if sentiment == "BEARISH_ALERT" and is_ready:
            is_ready = False
            notes.append("Veto de Noticias: CryptoPanic reporta flujo negativo importante.")

        s_filter = symbol_filters.get(sym, {"step_size": 0.01, "min_qty": 0.01, "min_notional": 10.0})
        sizing = compute_recommended_position(
            capital_total_usdt=capital_total_usdt,
            free_usdt=free_usdt_remaining,
            risk_pct=req.risk_pct,
            available_slots=max(1, slots_left_for_sizing) if slots_left_for_sizing > 0 else 0,
            entry_price=entry_price,
            stop_loss=stop_loss,
            symbol_filter=s_filter,
        )
        notes.append(str(sizing["note"]))

        if is_ready and sizing["min_notional_ok"] and slots_left_for_sizing > 0:
            slots_left_for_sizing -= 1
            free_usdt_remaining = max(
                0.0, free_usdt_remaining - float(sizing["position_size_usdt"]) * 1.001
            )

        candidates.append(
            SignalCandidate(
                symbol=sym,
                signal_status="SIGNAL_READY" if is_ready else "WATCHLIST",
                candle_close_utc=snap["candle_close_utc"],
                valid_until_utc=snap["valid_until_utc"],
                entry_price=round(entry_price, 6),
                max_chase_price=round(max_chase_price, 6),
                stop_loss=round(stop_loss, 6),
                take_profit=round(take_profit, 6),
                atr_4h=round(atr_val, 6),
                atr_pct=round(float(snap["atr_pct"]) * 100.0, 3),
                rsi_4h=round(float(snap["rsi_4h"]), 2),
                ema_fast_4h=round(float(snap["ema_fast_4h"]), 6),
                ema_slow_4h=round(float(snap["ema_slow_4h"]), 6),
                ema_macro_1d=round(float(snap["ema_macro_1d"]), 6),
                relative_strength_score=round(float(snap["rs_score"]), 3),
                position_size_usdt=float(sizing["position_size_usdt"]),
                position_qty=float(sizing["position_qty"]),
                risk_usdt=float(sizing["risk_usdt"]),
                effective_stop_pct=float(sizing["effective_stop_pct"]),
                min_notional_ok=bool(sizing["min_notional_ok"]),
                lot_step_size=float(sizing["lot_step_size"]),
                news_sentiment=sentiment,
                notes=notes,
            )
        )

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return ScanResponse(
        scan_id=str(uuid.uuid4()),
        scanned_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        elapsed_ms=elapsed_ms,
        manifest_hash=manifest.config_hash,
        strategy_approved=manifest.strategy_approved,
        execution_mode=manifest.execution_mode,
        usdt_mxn_rate=round(usdt_mxn_rate, 4),
        capital_total_usdt=capital_total_usdt,
        free_usdt=round(req.free_usdt, 2),
        available_slots=available_slots,
        fear_greed_value=fg_val,
        fear_greed_label=fg_label,
        circuit_breaker_active=circuit_breaker_active,
        circuit_breaker_reason=circuit_breaker_reason,
        candidates=candidates,
    )
