"""
Main entry point for the E8 Mind Kaleidoscope server.

This bootstraps the orchestrator from the modular package and wires the
HTTP/WebSocket endpoints using the existing legacy handlers for now.

Run: python main.py
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import os
import sys
import time
from pathlib import Path
import argparse
from types import ModuleType
from typing import Any


def _patch_env_defaults(cli_args: argparse.Namespace | None = None) -> None:
    # Make server non-interactive and safer by default
    os.environ.setdefault("E8_NON_INTERACTIVE", "1")
    os.environ.setdefault("E8_DISABLE_HEAVY_SUBSYSTEMS", "1")
    os.environ.setdefault("E8_PROVIDER", os.getenv("E8_PROVIDER", "stub"))
    if cli_args is not None:
        if getattr(cli_args, "non_interactive", None) is True:
            os.environ["E8_NON_INTERACTIVE"] = "1"
        if getattr(cli_args, "max_steps", None) is not None:
            os.environ["E8_MAX_STEPS"] = str(cli_args.max_steps)
        if getattr(cli_args, "port", None) is not None:
            os.environ["E8_PORT"] = str(cli_args.port)


def _load_legacy_server_module() -> ModuleType:
    """Load the legacy server as a module so we can reuse handlers."""
    root = Path(__file__).parent
    fname_candidates = [
        root / "e8_mind_server_M24.2.py",
        root / "e8_mind_server_M24.1.py",
    ]
    for f in fname_candidates:
        if f.exists():
            spec = importlib.util.spec_from_file_location(f.stem, str(f))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[f.stem] = mod
                # Prevent any blocking input during import
                _orig_input = __builtins__.get("input") if isinstance(__builtins__, dict) else getattr(__builtins__, "input", None)
                try:
                    def _no_input(prompt: str = "") -> str:
                        return ""
                    if isinstance(__builtins__, dict):
                        __builtins__["input"] = _no_input
                    else:
                        setattr(__builtins__, "input", _no_input)
                    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
                finally:
                    if _orig_input is not None:
                        if isinstance(__builtins__, dict):
                            __builtins__["input"] = _orig_input
                        else:
                            setattr(__builtins__, "input", _orig_input)
                return mod
    raise ImportError("Unable to find legacy server file e8_mind_server_M24.[1|2].py")


async def _run_server_and_cycle(cli_args: argparse.Namespace | None = None) -> None:
    _patch_env_defaults(cli_args)

    # Console from rich is optional; the factory can create its own if None
    try:
        from rich.console import Console
        # Enable recording so console export features in the legacy mind work
        console = Console(record=True)
    except Exception:
        console = None  # type: ignore

    # Import orchestrator from modular package
    from e8_mind.core.mind import new_default_mind

    # Build the mind with safe defaults
    mind: Any = new_default_mind(console=console)

    # Prefer modular handlers; fall back to legacy module if needed
    modular_http = None
    try:
        from e8_mind.http import handlers as modular_http  # type: ignore
    except Exception:
        modular_http = None
    legacy = None
    if modular_http is None:
        legacy = _load_legacy_server_module()

    # aiohttp is required for the server piece; if missing, run headless
    try:
        import aiohttp  # noqa: F401
        from aiohttp import web
    except Exception:
        if console:
            console.log("[bold yellow]aiohttp not installed. Running mind headlessly (no server).[/bold yellow]")
        # Run just a short cycle and exit
        await mind.run_cognitive_cycle(max_steps=int(os.getenv("E8_MAX_STEPS", "120")))
        return

    # Build the app and wire middleware and routes
    app = web.Application()

    # Basic CORS fallback middleware (compatible with absence of aiohttp_cors)
    try:
        import aiohttp_cors  # type: ignore
    except Exception:
        aiohttp_cors = None  # type: ignore

    @web.middleware
    async def _simple_cors_middleware(request, handler):
        resp = None
        if aiohttp_cors is None:
            if request.method == 'OPTIONS':
                resp = web.Response(status=204)
                resp.headers['Access-Control-Allow-Origin'] = '*'
                resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
                resp.headers['Access-Control-Allow-Headers'] = '*'
                return resp
        resp = await handler(request)
        if aiohttp_cors is None:
            try:
                resp.headers.setdefault('Access-Control-Allow-Origin', '*')
                resp.headers.setdefault('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
                resp.headers.setdefault('Access-Control-Allow-Headers', '*')
            except Exception:
                pass
        return resp

    # Connection rate limiter to avoid reconnect floods
    _conn_tracker: dict[str, float] = {}

    @web.middleware
    async def _rate_limiter(request, handler):
        if '/ws' in request.path or '/telemetry' in request.path:
            client_ip = request.remote or 'unknown'
            now = time.time()
            # cleanup
            for ip, last in list(_conn_tracker.items()):
                if now - last > 30:
                    _conn_tracker.pop(ip, None)
            # limit
            if client_ip in _conn_tracker and now - _conn_tracker[client_ip] < 1.0:
                if console:
                    console.log(f"[RATE LIMIT] Blocking rapid connection from {client_ip}")
                return web.Response(text='Rate limited', status=429)
            _conn_tracker[client_ip] = now
        return await handler(request)

    app.middlewares.append(_simple_cors_middleware)
    app.middlewares.append(_rate_limiter)

    # Share state with handlers
    app['mind'] = mind
    app['sse_clients'] = set()
    mind.sse_clients = app['sse_clients']
    app['ws_clients'] = set()
    mind.ws_clients = app['ws_clients']

    # Lightweight health endpoint for readiness checks
    async def _healthz(_request):
        return web.json_response({"status": "ok"})
    app.router.add_get('/healthz', _healthz)

    # Resolve handler callables from modular package when available, else legacy
    def handle(name: str):
        if modular_http is not None and hasattr(modular_http, name):
            return getattr(modular_http, name)
        if legacy is None:
            # If modular import partially failed, ensure legacy is loaded
            try:
                mod = _load_legacy_server_module()
            except Exception as e:
                raise
            return getattr(mod, name)
        return getattr(legacy, name)

    # Register routes (mirrors legacy)
    app.router.add_get("/api/memory/search", handle("handle_memory_search"))
    app.router.add_get("/api/state", handle("handle_get_state"))
    app.router.add_post("/api/action/dream", handle("handle_trigger_dream"))
    app.router.add_get("/api/qeng/telemetry", handle("handle_get_qeng_telemetry"))
    app.router.add_get("/api/qeng/ablation", handle("handle_get_qeng_ablation"))
    app.router.add_get("/api/qeng/probabilities", handle("handle_get_qeng_probabilities"))
    app.router.add_get("/metrics/summary", handle("handle_get_metrics_summary"))
    app.router.add_get("/metrics/live", handle("handle_get_metrics_live"))
    app.router.add_post("/quantizer", handle("handle_post_quantizer"))
    app.router.add_post("/snapshot", handle("handle_post_snapshot"))
    app.router.add_get("/api/telemetry", handle("handle_get_telemetry"))
    app.router.add_get("/api/blueprint", handle("handle_get_blueprint"))
    app.router.add_get("/api/telemetry/stream", handle("handle_stream_telemetry"))
    app.router.add_get("/api/telemetry/latest", handle("handle_get_telemetry"))
    app.router.add_get("/api/telemetry/ws", handle("handle_ws_telemetry"))
    app.router.add_get("/ws/telemetry", handle("handle_ws_telemetry"))
    app.router.add_get("/ws", handle("handle_ws_telemetry"))
    app.router.add_get("/api/graph", handle("handle_get_graph"))
    app.router.add_get("/api/graph/summary", handle("handle_get_graph_summary"))
    app.router.add_get("/api/node/{node_id}", handle("handle_get_node"))
    app.router.add_get("/api/lattice", handle("handle_get_lattice"))
    app.router.add_get("/api/bh/panel", handle("handle_get_bh_panel"))
    app.router.add_get("/api/metrics/recent", handle("handle_get_metrics_recent"))
    app.router.add_post("/api/concept/add", handle("handle_add_concept"))
    app.router.add_post("/api/concept", handle("handle_add_concept_legacy"))
    app.router.add_get("/", handle("handle_index"))

    # Static assets served at '/'
    static_path = str(Path(__file__).parent / 'static')
    if os.path.exists(static_path):
        try:
            app.router.add_static('/', static_path, show_index=True)
        except Exception:
            pass

    # Optional CORS library; if present, wrap routes
    if aiohttp_cors is not None:  # type: ignore
        cors = aiohttp_cors.setup(app, defaults={
            "*": aiohttp_cors.ResourceOptions(allow_credentials=True, expose_headers="*", allow_headers="*")  # type: ignore[attr-defined]
        })
        for route in list(app.router.routes()):
            try:
                cors.add(route)  # type: ignore[attr-defined]
            except Exception:
                pass

    # Graceful shutdown hooks from modular http or legacy
    for hook_name in ("shutdown_sse", "shutdown_market_feed", "shutdown_ws"):
        try:
            hook = None
            if modular_http is not None:
                hook = getattr(modular_http, hook_name, None)
            if hook is None and legacy is not None:
                hook = getattr(legacy, hook_name, None)
            if hook is not None:
                app.on_shutdown.append(hook)
        except Exception:
            pass

    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv('E8_PORT', '7871'))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()

    if console:
        console.log(f"[bold green]E8 Mind Server running at http://localhost:{port}[/bold green]")

    # Start background tasks
    try:
        curvature_task = asyncio.create_task(mind._run_curvature_stream())
    except Exception:
        curvature_task = None

    # If running in server-only mode, skip the cognitive cycle and keep the server alive
    if cli_args is not None and getattr(cli_args, "server_only", False):
        if console:
            console.log("[bold cyan]Server-only mode: cognitive cycle disabled. Press Ctrl+C to stop.[/bold cyan]")
        try:
            # Sleep indefinitely until cancelled (e.g., Ctrl+C)
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if console:
                console.log("[bold yellow]Server-only mode cancelled; keeping server up for linger window...[/bold yellow]")
            linger = int(os.getenv("E8_LINGER_SECS", "30"))
            await asyncio.sleep(max(0, linger))
        finally:
            try:
                if curvature_task:
                    curvature_task.cancel()
                    with contextlib.suppress(Exception):
                        await curvature_task
            except Exception:
                pass
            with contextlib.suppress(Exception):
                await runner.cleanup()
        return

    max_steps_env = os.getenv("E8_MAX_STEPS", "")
    try:
        if max_steps_env:
            max_steps = int(max_steps_env)
        else:
            # Use a safer default in non-interactive/dev sessions
            max_steps = 300 if os.getenv("E8_NON_INTERACTIVE", "0") == "1" else 297600
    except ValueError:
        max_steps = 300 if os.getenv("E8_NON_INTERACTIVE", "0") == "1" else 297600

    if console:
        console.log(f"[cyan]Launching cognitive cycle for max_steps={max_steps}[/cyan]")

    cycle_task = asyncio.create_task(mind.run_cognitive_cycle(max_steps=max_steps))
    try:
        await cycle_task
        # Even on normal completion, keep the server alive for a configurable linger window
        if console:
            console.log("[bold green]Cognitive cycle complete; keeping server up for linger window...[/bold green]")
        linger = int(os.getenv("E8_LINGER_SECS", "30"))
        await asyncio.sleep(max(0, linger))
    except asyncio.CancelledError:
        if console:
            console.log("[bold yellow]Cognitive cycle cancelled; keeping server up for linger window...[/bold yellow]")
        # Keep server alive for a configurable linger window so tests/clients can complete
        linger = int(os.getenv("E8_LINGER_SECS", "30"))
        await asyncio.sleep(max(0, linger))
    except Exception as e:
        # Unexpected error in cycle; keep the server up for linger window so tests can still probe endpoints
        if console:
            console.log(f"[bold red]Cognitive cycle error:[/bold red] {e}; keeping server up for linger window...")
        linger = int(os.getenv("E8_LINGER_SECS", "30"))
        await asyncio.sleep(max(0, linger))
    finally:
        # Stop background task
        try:
            if curvature_task:
                curvature_task.cancel()
                with contextlib.suppress(Exception):
                    await curvature_task
        except Exception:
            pass
        # Cleanup server runner to free the port
        with contextlib.suppress(Exception):
            await runner.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description="E8 Mind Kaleidoscope Server")
    parser.add_argument("--port", type=int, help="Port to bind the HTTP server (default from E8_PORT or 7871)")
    parser.add_argument("--max-steps", dest="max_steps", type=int, help="Max steps for the cognitive cycle (default from E8_MAX_STEPS or 300 in non-interactive)")
    parser.add_argument("--non-interactive", action="store_true", help="Force non-interactive mode")
    parser.add_argument("--server-only", action="store_true", help="Start HTTP server without running cognitive cycle (press Ctrl+C to stop)")
    args = parser.parse_args()
    asyncio.run(_run_server_and_cycle(args))


if __name__ == "__main__":
    main()
