# Programa 2: `crypto-signal-api` (Microservicio Cloud de Escaneo y Evaluación)

Microservicio **100% sin estado (stateless)** en FastAPI diseñado para ejecutarse en Render Free (región Frankfurt). Consume el archivo `manifests/strategy_manifest.json` certificado por el Programa 1 (`crypto-quant-lab`), descarga en paralelo (`asyncio` + `httpx`) desde `data-api.binance.vision` y expone endpoints REST protegidos por `X-API-Key` para el cliente Android (`crypto-swing-mobile`).

## Endpoints (`X-API-Key` requerido excepto `/api/v5/health`)
- `GET /api/v5/health`: Despierta el contenedor (Cold Start) y verifica el `config_hash` cargado.
- `GET /api/v5/manifest`: Devuelve el manifiesto estratégico activo.
- `PUT /api/v5/manifest`: Permite actualizar el `strategy_manifest.json` remotamente.
- `POST /api/v5/scan`: Escanea los 10 activos en paralelo, convierte `capital_total_mxn` a USDT, valida `LOT_SIZE`/`minNotional` e incluye fricción del `0.36%`.
- `GET /api/v5/scan/{scan_id}`: Recupera un escaneo reciente por su ID.
- `POST /api/v5/evaluate`: Evalúa posiciones abiertas con resolución intravela pesimista (`STOP_HIT` vs `TP_HIT`).

## Ejecución Local
```bash
pip install -r requirements.txt
pytest -v
uvicorn app.main:app --reload --port 8000
```
