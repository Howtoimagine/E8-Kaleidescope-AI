# Upgrade Notes: M24 → Modular Edition

These notes help you migrate from the legacy M24 monolith to the Modular Edition introduced in this branch.

## What changed

- New entrypoint: `main.py` (aiohttp server + orchestrator bootstrap)
- Modular packages:
  - `e8_mind/physics/*`
  - `e8_mind/memory/*`
  - `e8_mind/cognitive/*`
  - `e8_mind/core/*` (orchestrator/proxy)
  - `e8_mind/http/*` (HTTP & WS handlers)
- Health/readiness endpoint: `GET /healthz`
- Linger behavior after run completion (configurable via `E8_LINGER_SECS`)
- Optional server-only mode for stable endpoint testing (`--server-only`)

Legacy servers (`e8_mind_server_M24.1.py` / `.2.py`) still exist for parity/fallback. New code should import from `e8_mind.*` or `e8_mind.monolith`.

## Run the new server

Windows PowerShell examples:

```powershell
# 1) Server-only (no cognitive cycle) – ideal for UI/smoke tests
python .\main.py --server-only --non-interactive --port 7871

# 2) Full run (finite steps) with linger window
$env:E8_LINGER_SECS='120'
python .\main.py --non-interactive --port 7871 --max-steps 600

# 3) Health check
curl http://localhost:7871/healthz

# 4) Endpoint smoke tests
python .\test_endpoints.py --port 7871
```

## Endpoint parity

- Lattice: response includes legacy-compatible keys `roots_3d` and `active_highlights`.
- Graph, telemetry, blueprint, node, metrics, and BH panel endpoints are wired via `e8_mind/http/handlers.py` with legacy fallbacks.
- SSE and WebSocket streams support graceful shutdown hooks.

## Environment & CLI

Environment defaults (override as needed):

- `E8_NON_INTERACTIVE=1` – avoid interactive prompts
- `E8_DISABLE_HEAVY_SUBSYSTEMS=0` – set to `1` for lighter runs
- `E8_MAX_STEPS` – max cognitive steps (or `--max-steps` CLI)
- `E8_PORT` – default port if `--port` not provided
- `E8_LINGER_SECS` – seconds to keep server alive after run completion

CLI flags in `main.py`:

- `--port <int>`
- `--max-steps <int>`
- `--non-interactive`
- `--server-only`

## Windows tips

- Set env vars in PowerShell as `$env:NAME='value'` (no `export`).
- If a port is in use, choose another `--port` or wait for linger to release the socket.

## Breaking changes

- Entry point is now `main.py` (instead of the monolith). Legacy scripts can still be used, but are no longer the preferred path.
- HTTP handlers have moved to `e8_mind/http/handlers.py`. If you previously extended endpoints in the monolith, port them here.

## Migration checklist

- [ ] Start the new server: `python .\main.py --server-only --port 7871`
- [ ] Verify `/healthz` returns `ok`
- [ ] Run `python .\test_endpoints.py --port 7871`
- [ ] Update any imports to `e8_mind.*` or `e8_mind.monolith`
- [ ] If you customized handlers, migrate logic into `e8_mind/http/handlers.py`
- [ ] Optionally tune `E8_LINGER_SECS` for your workflow

## FAQ

- Q: Can I still use M24.1/M24.2?  
  A: Yes. They’re kept for parity; `main.py` prefers modular handlers and proxies to legacy classes when needed.

- Q: Why server-only mode?  
  A: It keeps the HTTP server stable without the cognitive cycle, ideal for UI and contract tests.

- Q: Do I need aiohttp-cors?  
  A: The server provides a simple CORS fallback; `aiohttp_cors` is optional.
