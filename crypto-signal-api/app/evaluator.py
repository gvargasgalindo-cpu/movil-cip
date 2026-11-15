"""Evaluador automático de posiciones abiertas con resolución intravela pesimista (SDD 2.4)."""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List

import httpx
import pandas as pd

from .models import (
    EvaluatedPositionResult,
    EvaluateRequest,
    EvaluateResponse,
    OpenPositionToEvaluate,
    StrategyParameters,
)
from .recommender import BINANCE_VISION_BASE

ENTRY_FEE_PCT = 0.0010
EXIT_FEE_PCT = 0.0010
ENTRY_SLIPPAGE_PCT = 0.0008
EXIT_SLIPPAGE_PCT = 0.0008


async def _fetch_klines_since(
    client: httpx.AsyncClient, symbol: str, start_ms: int
) -> pd.DataFrame:
    resp = await client.get(
        f"{BINANCE_VISION_BASE}/klines",
        params={
            "symbol": symbol,
            "interval": "4h",
            "startTime": start_ms,
            "limit": 500,
        },
        timeout=3.0,
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
    )[["open_time", "open", "high", "low", "close", "close_time"]]

    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"].astype("int64"), unit="ms", utc=True)
    return df


def _compute_net_pnl_and_r(
    entry_raw: float,
    exit_raw: float,
    stop_loss: float,
    qty: float,
) -> tuple[float, float]:
    """Calcula PnL neto y múltiplo R neto descontando el 0.36% de fricción round-trip."""
    entry_eff = entry_raw * (1.0 + ENTRY_SLIPPAGE_PCT)
    exit_eff = exit_raw * (1.0 - EXIT_SLIPPAGE_PCT)
    stop_eff = stop_loss * (1.0 - EXIT_SLIPPAGE_PCT)

    notional_in = qty * entry_eff * (1.0 + ENTRY_FEE_PCT)
    notional_out = qty * exit_eff * (1.0 - EXIT_FEE_PCT)
    net_pnl = notional_out - notional_in

    notional_stop = qty * stop_eff * (1.0 - EXIT_FEE_PCT)
    initial_risk_usdt = max(1e-6, notional_in - notional_stop)
    r_mult = net_pnl / initial_risk_usdt
    return round(net_pnl, 2), round(r_mult, 3)


def evaluate_single_position_on_df(
    pos: OpenPositionToEvaluate,
    df_4h: pd.DataFrame,
    params: StrategyParameters,
) -> EvaluatedPositionResult:
    """
    Evalúa una posición abierta sobre las velas 4H posteriores a opened_at_utc.
    Aplica resolución intravela pesimista (SDD 1.3.4):
      Si low <= stop_loss y high >= take_profit en la misma vela -> STOP_HIT.
    """
    opened_dt = pd.Timestamp(pos.opened_at_utc)
    if opened_dt.tzinfo is None:
        opened_dt = opened_dt.tz_localize("UTC")
    else:
        opened_dt = opened_dt.tz_convert("UTC")

    if df_4h.empty:
        return EvaluatedPositionResult(
            position_id=pos.position_id,
            symbol=pos.symbol,
            status="OPEN",
            resolved_at_utc=None,
            exit_price=None,
            current_price=pos.entry_price,
            bars_elapsed_4h=0,
            net_pnl_usdt=0.0,
            r_multiple_net=0.0,
            notes="Sin velas posteriores disponibles aún.",
        )

    # Filtramos únicamente las velas cuyo cierre ocurrió después de la apertura de la posición
    sub = df_4h[df_4h["close_time"] > opened_dt].sort_values("open_time").reset_index(drop=True)
    if sub.empty:
        last_close = float(df_4h["close"].iloc[-1])
        net_pnl, r_net = _compute_net_pnl_and_r(
            pos.entry_price, last_close, pos.stop_loss, pos.qty
        )
        return EvaluatedPositionResult(
            position_id=pos.position_id,
            symbol=pos.symbol,
            status="OPEN",
            resolved_at_utc=None,
            exit_price=None,
            current_price=last_close,
            bars_elapsed_4h=0,
            net_pnl_usdt=net_pnl,
            r_multiple_net=r_net,
            notes="En vela inicial; sin resolución aún.",
        )

    bars_elapsed = 0
    for _, row in sub.iterrows():
        bars_elapsed += 1
        low_val = float(row["low"])
        high_val = float(row["high"])
        close_val = float(row["close"])
        close_ts_iso = pd.Timestamp(row["close_time"]).strftime("%Y-%m-%dT%H:%M:%SZ")

        hit_sl = low_val <= pos.stop_loss
        hit_tp = high_val >= pos.take_profit

        # Regla SDD 1.3.4: Resolución intravela pesimista
        if hit_sl and hit_tp:
            net_pnl, r_net = _compute_net_pnl_and_r(
                pos.entry_price, pos.stop_loss, pos.stop_loss, pos.qty
            )
            return EvaluatedPositionResult(
                position_id=pos.position_id,
                symbol=pos.symbol,
                status="STOP_HIT",
                resolved_at_utc=close_ts_iso,
                exit_price=pos.stop_loss,
                current_price=close_val,
                bars_elapsed_4h=bars_elapsed,
                net_pnl_usdt=net_pnl,
                r_multiple_net=r_net,
                notes="Resolución intravela pesimista: la vela tocó SL y TP simultáneamente -> STOP_HIT.",
            )
        if hit_sl:
            net_pnl, r_net = _compute_net_pnl_and_r(
                pos.entry_price, pos.stop_loss, pos.stop_loss, pos.qty
            )
            return EvaluatedPositionResult(
                position_id=pos.position_id,
                symbol=pos.symbol,
                status="STOP_HIT",
                resolved_at_utc=close_ts_iso,
                exit_price=pos.stop_loss,
                current_price=close_val,
                bars_elapsed_4h=bars_elapsed,
                net_pnl_usdt=net_pnl,
                r_multiple_net=r_net,
                notes=f"Stop Loss alcanzado en vela #{bars_elapsed} (low={low_val}).",
            )
        if hit_tp:
            net_pnl, r_net = _compute_net_pnl_and_r(
                pos.entry_price, pos.take_profit, pos.stop_loss, pos.qty
            )
            return EvaluatedPositionResult(
                position_id=pos.position_id,
                symbol=pos.symbol,
                status="TP_HIT",
                resolved_at_utc=close_ts_iso,
                exit_price=pos.take_profit,
                current_price=close_val,
                bars_elapsed_4h=bars_elapsed,
                net_pnl_usdt=net_pnl,
                r_multiple_net=r_net,
                notes=f"Take Profit alcanzado en vela #{bars_elapsed} (high={high_val}).",
            )
        if bars_elapsed >= params.time_stop_bars_4h:
            net_pnl, r_net = _compute_net_pnl_and_r(
                pos.entry_price, close_val, pos.stop_loss, pos.qty
            )
            return EvaluatedPositionResult(
                position_id=pos.position_id,
                symbol=pos.symbol,
                status="TIME_EXIT",
                resolved_at_utc=close_ts_iso,
                exit_price=close_val,
                current_price=close_val,
                bars_elapsed_4h=bars_elapsed,
                net_pnl_usdt=net_pnl,
                r_multiple_net=r_net,
                notes=f"Time Stop alcanzado ({bars_elapsed} velas 4H sin tocar SL ni TP).",
            )

    current_price = float(sub["close"].iloc[-1])
    net_pnl, r_net = _compute_net_pnl_and_r(
        pos.entry_price, current_price, pos.stop_loss, pos.qty
    )
    return EvaluatedPositionResult(
        position_id=pos.position_id,
        symbol=pos.symbol,
        status="OPEN",
        resolved_at_utc=None,
        exit_price=None,
        current_price=current_price,
        bars_elapsed_4h=bars_elapsed,
        net_pnl_usdt=net_pnl,
        r_multiple_net=r_net,
        notes=f"Posición activa ({bars_elapsed}/{params.time_stop_bars_4h} velas 4H).",
    )


async def evaluate_open_positions(
    req: EvaluateRequest,
    params: StrategyParameters,
) -> EvaluateResponse:
    t0 = time.perf_counter()
    if not req.positions:
        return EvaluateResponse(
            evaluated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            elapsed_ms=0,
            results=[],
        )

    # Agrupamos por símbolo y encontramos el menor opened_at_utc por símbolo para hacer 1 sola petición por moneda
    symbol_min_ms: Dict[str, int] = {}
    for p in req.positions:
        dt = pd.Timestamp(p.opened_at_utc)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        ms = max(0, int(dt.timestamp() * 1000) - 4 * 3600 * 1000)
        if p.symbol not in symbol_min_ms or ms < symbol_min_ms[p.symbol]:
            symbol_min_ms[p.symbol] = ms

    unique_symbols = list(symbol_min_ms.keys())
    async with httpx.AsyncClient(headers={"User-Agent": "crypto-signal-api/5.0"}) as client:
        tasks = [_fetch_klines_since(client, sym, symbol_min_ms[sym]) for sym in unique_symbols]
        fetched = await asyncio.gather(*tasks, return_exceptions=True)

    symbol_dfs: Dict[str, pd.DataFrame] = {}
    for sym, res in zip(unique_symbols, fetched):
        symbol_dfs[sym] = pd.DataFrame() if isinstance(res, Exception) else res

    results: List[EvaluatedPositionResult] = []
    for p in req.positions:
        df_4h = symbol_dfs.get(p.symbol, pd.DataFrame())
        results.append(evaluate_single_position_on_df(p, df_4h, params))

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return EvaluateResponse(
        evaluated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        elapsed_ms=elapsed_ms,
        results=results,
    )
