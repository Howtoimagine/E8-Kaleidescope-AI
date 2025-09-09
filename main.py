from __future__ import annotations

"""
E8Mind modular entry point.

Usage:
  python main.py orchestrator --steps 100 --resume
  python main.py server --host 0.0.0.0 --port 8080

Env overrides are read via core.config.AppConfig and config/settings.py.
"""

import argparse
import os
import sys
from typing import Optional

from config.settings import load_settings
from core.config import AppConfig

def run_orchestrator(args: argparse.Namespace, cfg: AppConfig) -> None:
    from runtime.orchestrator import Orchestrator
    orch = Orchestrator(cfg)
    orch.start(run_name=args.run_name, resume=args.resume)
    orch.run(max_steps=int(args.steps), label_hint=args.label_hint)

def run_server(args: argparse.Namespace, cfg: AppConfig) -> None:
    import uvicorn
    from web.server import create_app

    host = args.host or cfg.web_host
    port = int(args.port or cfg.web_port)

    app = create_app(cfg)
    uvicorn.run(app, host=host, port=port, reload=args.reload)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="E8Mind")
    sub = p.add_subparsers(dest="command", required=True)

    # Orchestrator
    po = sub.add_parser("orchestrator", help="Run runtime orchestrator loop")
    po.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    po.add_argument("--resume", action="store_true", help="Resume from latest run")
    po.add_argument("--run-name", type=str, default=None, help="Explicit run folder name")
    po.add_argument("--label-hint", type=str, default=None, help="Optional hint label for seed synthesis")

    # Server
    ps = sub.add_parser("server", help="Run FastAPI server")
    ps.add_argument("--host", type=str, default=None, help="Host to bind (default from AppConfig)")
    ps.add_argument("--port", type=int, default=None, help="Port to bind (default from AppConfig)")
    ps.add_argument("--reload", action="store_true", help="Enable autoreload for development")

    return p

def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = load_settings()

    if args.command == "orchestrator":
        run_orchestrator(args, cfg)
    elif args.command == "server":
        run_server(args, cfg)
    else:
        parser.error(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()
