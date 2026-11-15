"""Microservicio Cloud FastAPI Stateless (Programa 2: crypto-signal-api - SDD v5)."""
from __future__ import annotations

import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from .evaluator import evaluate_open_positions
from .models import (
    EvaluateRequest,
    EvaluateResponse,
    ScanRequest,
    ScanResponse,
    StrategyManifest,
)
from .scanner import execute_market_scan

BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BASE_DIR / "manifests" / "strategy_manifest.json"

# Buffer efímero en memoria solo para compatibilidad opcional con GET /api/v5/scan/{scan_id}
_RECENT_SCANS: OrderedDict[str, ScanResponse] = OrderedDict()

app = FastAPI(
    title="crypto-signal-api (SDD v5)",
    description="Microservicio Cloud 100% sin estado para escaneo 4H/1D y evaluación de posiciones.",
    version="5.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(x_api_key: str = Header(default="", alias="X-API-Key")) -> str:
    """Autenticación obligatoria por header X-API-Key (SDD 2.2)."""
    expected_key = os.getenv("SIGNAL_API_KEY", "dev-crypto-swing-v5-key")
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Header X-API-Key inválido o ausente.",
        )
    return x_api_key


def load_active_manifest() -> StrategyManifest:
    if not MANIFEST_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No se encontró manifests/strategy_manifest.json del Programa 1.",
        )
    try:
        raw = MANIFEST_PATH.read_text(encoding="utf-8")
        return StrategyManifest.model_validate_json(raw)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error validando strategy_manifest.json: {exc}",
        ) from exc


@app.get("/api/v5/health")
async def health_check() -> Dict[str, str | bool]:
    """Endpoint de despertar (Cold Start en Render Free) y verificación de manifiesto."""
    manifest_exists = MANIFEST_PATH.exists()
    config_hash = "none"
    approved = False
    mode = "UNKNOWN"
    if manifest_exists:
        try:
            m = load_active_manifest()
            config_hash = m.config_hash
            approved = m.strategy_approved
            mode = m.execution_mode
        except Exception:
            pass
    return {
        "status": "ok",
        "service": "crypto-signal-api",
        "version": "5.0",
        "manifest_loaded": manifest_exists,
        "config_hash": config_hash,
        "strategy_approved": approved,
        "execution_mode": mode,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@app.get("/api/v5/manifest", response_model=StrategyManifest)
async def get_manifest(_: str = Depends(verify_api_key)) -> StrategyManifest:
    """Devuelve el strategy_manifest.json activo."""
    return load_active_manifest()


@app.put("/api/v5/manifest", response_model=StrategyManifest)
async def update_manifest(
    new_manifest: StrategyManifest,
    _: str = Depends(verify_api_key),
) -> StrategyManifest:
    """Endpoint de administración para subir un nuevo strategy_manifest.json desde el Programa 1."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(new_manifest.model_dump_json(indent=2), encoding="utf-8")
    return new_manifest


@app.post("/api/v5/scan", response_model=ScanResponse)
async def post_scan(
    req: ScanRequest,
    _: str = Depends(verify_api_key),
) -> ScanResponse:
    """
    Escanea en paralelo (<2.5s) los 10 activos aprobados en strategy_manifest.json,
    calcula tamaño de posición en USDT a partir de capital_total_mxn y devuelve candidatos.
    """
    manifest = load_active_manifest()
    response = await execute_market_scan(manifest, req)

    _RECENT_SCANS[response.scan_id] = response
    while len(_RECENT_SCANS) > 25:
        _RECENT_SCANS.popitem(last=False)

    return response


@app.get("/api/v5/scan/{scan_id}", response_model=ScanResponse)
async def get_scan_by_id(
    scan_id: str,
    _: str = Depends(verify_api_key),
) -> ScanResponse:
    """Recupera el resultado de un escaneo reciente por su ID."""
    if scan_id not in _RECENT_SCANS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scan ID {scan_id} no encontrado (servidor stateless).",
        )
    return _RECENT_SCANS[scan_id]


@app.post("/api/v5/evaluate", response_model=EvaluateResponse)
async def post_evaluate(
    req: EvaluateRequest,
    _: str = Depends(verify_api_key),
) -> EvaluateResponse:
    """
    Evalúa posiciones abiertas contra velas 4H reales aplicando resolución intravela pesimista
    (STOP_HIT tiene prioridad si una vela toca SL y TP simultáneamente).
    """
    manifest = load_active_manifest()
    return await evaluate_open_positions(req, manifest.parameters)
