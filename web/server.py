from __future__ import annotations

import os
from typing import Any, List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import AppConfig
from physics.e8_lattice import E8Physics
from physics.engines import QuantumEngine, ClassicalEngine
from memory.manager import MemoryManager
from .middleware import LoggingMiddleware, APIKeyAuthMiddleware
from .handlers import build_router


class Console:
    def log(self, *args, **kwargs):
        print(*args)


class Services:
    """
    Simple DI container for the API.
    Attributes:
      - config: AppConfig
      - console: Console-like with .log
      - physics: E8Physics
      - qengine: QuantumEngine
      - cengine: ClassicalEngine
      - memory: MemoryManager
    """

    def __init__(self, config: AppConfig, console: Any,
                 physics: E8Physics, qengine: QuantumEngine,
                 cengine: ClassicalEngine, memory: MemoryManager) -> None:
        self.config = config
        self.console = console
        self.physics = physics
        self.qengine = qengine
        self.cengine = cengine
        self.memory = memory


def build_services(config: Optional[AppConfig] = None, console: Optional[Any] = None) -> Services:
    cfg = config or AppConfig.from_env()
    con = console or Console()
    physics = E8Physics(con)
    qengine = QuantumEngine(physics, config={"kernel": "cosine", "rbf_sigma": 0.8}, console=con)
    cengine = ClassicalEngine(physics, console=con)
    memory = MemoryManager(embed_dim=cfg.embed_dim, seed=cfg.global_seed)
    return Services(cfg, con, physics, qengine, cengine, memory)


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    cfg = config or AppConfig.from_env()
    app = FastAPI(title="E8Mind API", version="0.1.0")

    # CORS
    cors_origins = os.getenv("E8_CORS_ORIGINS", "*")
    if cors_origins == "*" or not cors_origins.strip():
        allowed_origins: List[str] = ["*"]
    else:
        allowed_origins = [o.strip() for o in cors_origins.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Logging + API Key auth
    app.add_middleware(LoggingMiddleware)
    # API key auth enabled only if E8_API_KEY is set
    app.add_middleware(APIKeyAuthMiddleware)

    # Build services and attach to app state
    services = build_services(cfg, Console())
    app.state.services = services

    # Root endpoint
    @app.get("/")
    def root():
        return {"name": "E8Mind", "status": "ok"}

    # API router
    router = build_router(services)
    app.include_router(router, prefix="/api")

    return app


# Module-level ASGI app for uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    cfg = AppConfig.from_env()
    uvicorn.run("web.server:app", host=cfg.web_host, port=cfg.web_port, reload=True)
