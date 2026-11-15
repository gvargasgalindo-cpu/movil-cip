"""Contratos de datos Pydantic entre Programa 1, Programa 2 y Programa 3 (SDD v5)."""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class StrategyParameters(BaseModel):
    ema_macro_1d: int = 200
    ema_fast_4h: int = 21
    ema_slow_4h: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    sl_atr_mult: float = 1.4
    tp_r_mult: float = 3.0
    rsi_entry_min: float = 55.0
    rsi_entry_max: float = 74.0
    pullback_lookback_bars: int = 6
    pullback_atr_tolerance: float = 0.60
    breakout_lookback_bars: int = 16
    volume_mult_min: float = 1.25
    min_atr_pct: float = 0.011
    max_atr_pct: float = 0.045
    time_stop_bars_4h: int = 48


class StrategyManifest(BaseModel):
    manifest_version: str = "5.0"
    generated_at_utc: str
    config_hash: str
    strategy_approved: bool
    execution_mode: str = "PAPER_ONLY"
    approved_universe: List[str]
    parameters: StrategyParameters
    validation_summary: Optional[Dict] = None
    rejection_reasons: List[str] = Field(default_factory=list)


class ScanRequest(BaseModel):
    capital_total_mxn: float = Field(..., gt=0, description="Capital total de la cartera expresado en MXN")
    free_usdt: float = Field(..., ge=0, description="Efectivo libre disponible en USDT para nuevas compras")
    risk_pct: float = Field(default=1.0, gt=0, le=5.0, description="Porcentaje de riesgo por operación (ej. 1.0 = 1%)")
    open_positions: List[str] = Field(default_factory=list, description="Símbolos actualmente abiertos (máx 3)")
    usdt_mxn_override: Optional[float] = Field(default=None, gt=0, description="Tipo de cambio manual opcional")
    include_watchlist: bool = Field(
        default=True,
        description="Si es True, devuelve también candidatos en tendencia alcista (WATCHLIST) además de SIGNAL_READY",
    )


class SignalCandidate(BaseModel):
    symbol: str
    signal_status: str = Field(description="SIGNAL_READY | WATCHLIST")
    candle_close_utc: str
    valid_until_utc: str
    entry_price: float
    max_chase_price: float
    stop_loss: float
    take_profit: float
    atr_4h: float
    atr_pct: float
    rsi_4h: float
    ema_fast_4h: float
    ema_slow_4h: float
    ema_macro_1d: float
    relative_strength_score: float
    position_size_usdt: float
    position_qty: float
    risk_usdt: float
    effective_stop_pct: float
    min_notional_ok: bool
    lot_step_size: float
    news_sentiment: str = "NEUTRAL"
    notes: List[str] = Field(default_factory=list)


class ScanResponse(BaseModel):
    scan_id: str
    scanned_at_utc: str
    elapsed_ms: int
    manifest_hash: str
    strategy_approved: bool
    execution_mode: str
    usdt_mxn_rate: float
    capital_total_usdt: float
    free_usdt: float
    available_slots: int
    fear_greed_value: int
    fear_greed_label: str
    circuit_breaker_active: bool
    circuit_breaker_reason: Optional[str] = None
    candidates: List[SignalCandidate]


class OpenPositionToEvaluate(BaseModel):
    position_id: str = Field(..., description="ID único local en SQLite del teléfono")
    symbol: str
    entry_price: float = Field(..., gt=0)
    stop_loss: float = Field(..., gt=0)
    take_profit: float = Field(..., gt=0)
    qty: float = Field(default=1.0, gt=0)
    opened_at_utc: str = Field(..., description="Timestamp ISO-8601 UTC de apertura de la operación")


class EvaluateRequest(BaseModel):
    positions: List[OpenPositionToEvaluate]


class EvaluatedPositionResult(BaseModel):
    position_id: str
    symbol: str
    status: str = Field(description="OPEN | TP_HIT | STOP_HIT | TIME_EXIT")
    resolved_at_utc: Optional[str] = None
    exit_price: Optional[float] = None
    current_price: float
    bars_elapsed_4h: int
    net_pnl_usdt: float
    r_multiple_net: float
    notes: str


class EvaluateResponse(BaseModel):
    evaluated_at_utc: str
    elapsed_ms: int
    results: List[EvaluatedPositionResult]
