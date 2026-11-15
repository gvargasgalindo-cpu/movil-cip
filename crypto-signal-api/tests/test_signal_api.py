"""Pruebas unitarias y de integración para crypto-signal-api (Programa 2 - SDD v5)."""
from __future__ import annotations

import asyncio
import pandas as pd
from fastapi.testclient import TestClient

from app.evaluator import evaluate_single_position_on_df
from app.main import app
from app.models import OpenPositionToEvaluate, StrategyParameters
from app.recommender import compute_recommended_position, quantize_lot_size

client = TestClient(app)
API_HEADERS = {"X-API-Key": "dev-crypto-swing-v5-key"}


def test_auth_required_on_protected_endpoints():
    r_no_key = client.get("/api/v5/manifest")
    assert r_no_key.status_code == 401

    r_ok = client.get("/api/v5/manifest", headers=API_HEADERS)
    assert r_ok.status_code == 200
    data = r_ok.json()
    assert data["manifest_version"] == "5.0"
    assert len(data["approved_universe"]) == 10


def test_health_endpoint_for_render_cold_start():
    r = client.get("/api/v5/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["manifest_loaded"] is True


def test_recommender_sizing_and_binance_lot_filters():
    # Capital = 100,000 MXN (~5,000 USDT a TC=20), riesgo = 1.0% (50 USDT)
    # Entry = 100.0, SL = 95.0 (5.0% raw stop + 0.36% fricción = 5.36% stop efectivo)
    # Notional ideal = 50 / 0.0536 = 932.83 USDT -> qty = 9.3283 -> step 0.01 -> 9.32
    s_filter = {"step_size": 0.01, "min_qty": 0.01, "min_notional": 10.0}
    res = compute_recommended_position(
        capital_total_usdt=5000.0,
        free_usdt=3000.0,
        risk_pct=1.0,
        available_slots=3,
        entry_price=100.0,
        stop_loss=95.0,
        symbol_filter=s_filter,
    )
    assert res["min_notional_ok"] is True
    assert res["position_qty"] == 9.32
    assert res["position_size_usdt"] == 932.0
    assert quantize_lot_size(12.3456, 0.01) == 12.34


def test_evaluator_pessimistic_intracandle_resolution():
    pos = OpenPositionToEvaluate(
        position_id="pos-001",
        symbol="DOGEUSDT",
        entry_price=0.20,
        stop_loss=0.18,
        take_profit=0.26,
        qty=1000.0,
        opened_at_utc="2026-11-10T00:00:00Z",
    )
    # Vela posterior que toca tanto SL (0.17 <= 0.18) como TP (0.27 >= 0.26)
    df_4h = pd.DataFrame(
        {
            "open_time": [pd.Timestamp("2026-11-10T04:00:00Z")],
            "open": [0.20],
            "high": [0.27],
            "low": [0.17],
            "close": [0.25],
            "close_time": [pd.Timestamp("2026-11-10T07:59:59Z")],
        }
    )
    params = StrategyParameters()
    eval_res = evaluate_single_position_on_df(pos, df_4h, params)
    assert eval_res.status == "STOP_HIT"
    assert eval_res.exit_price == 0.18
    assert eval_res.net_pnl_usdt < 0


def test_live_parallel_scan_and_get_by_id():
    payload = {
        "capital_total_mxn": 40000.0,
        "free_usdt": 1500.0,
        "risk_pct": 1.0,
        "open_positions": ["BTCUSDT"],
        "usdt_mxn_override": 20.0,
        "include_watchlist": True,
    }
    resp = client.post("/api/v5/scan", json=payload, headers=API_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["capital_total_usdt"] == 2000.0
    assert data["available_slots"] == 2
    assert isinstance(data["candidates"], list)

    scan_id = data["scan_id"]
    resp_get = client.get(f"/api/v5/scan/{scan_id}", headers=API_HEADERS)
    assert resp_get.status_code == 200
    assert resp_get.json()["scan_id"] == scan_id
