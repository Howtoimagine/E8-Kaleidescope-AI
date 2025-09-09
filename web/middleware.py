import os
import time
import logging
from typing import Optional, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Simple request logging middleware with timing.
    Logs: METHOD PATH STATUS DURATION_MS
    """

    def __init__(self, app, logger_name: str = "e8mind.web"):
        super().__init__(app)
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
            status = getattr(response, "status_code", 200)
        except Exception as exc:
            status = 500
            self.logger.exception("Unhandled error for %s %s", request.method, request.url.path)
            return JSONResponse({"detail": "Internal Server Error"}, status_code=500)
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self.logger.info("%s %s %s %.2fms", request.method, request.url.path, status, duration_ms)
        return response


class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    Lightweight API key auth.
    - Reads expected key from constructor or env E8_API_KEY
    - Accepts via header 'X-API-Key' or query parameter 'api_key'
    - Skips auth when no expected key is configured
    - Exempts common public paths (health, docs, openapi)
    """

    def __init__(
        self,
        app,
        expected_api_key: Optional[str] = None,
        header_name: str = "X-API-Key",
        query_param: str = "api_key",
        exempt_paths: Optional[Set[str]] = None,
    ):
        super().__init__(app)
        env_key = os.getenv("E8_API_KEY", "").strip()
        self.expected_api_key = (expected_api_key or env_key).strip()
        self.header_name = header_name
        self.query_param = query_param
        self.exempt_paths = exempt_paths or {
            "/",
            "/health",
            "/api/health",
            "/openapi.json",
            "/docs",
            "/docs/oauth2-redirect",
            "/redoc",
        }
        self.enabled = bool(self.expected_api_key)

    def _is_exempt(self, path: str) -> bool:
        if path in self.exempt_paths:
            return True
        # allow docs subpaths
        return path.startswith("/docs")

    async def dispatch(self, request: Request, call_next):
        if not self.enabled or self._is_exempt(request.url.path):
            return await call_next(request)

        provided = request.headers.get(self.header_name) or request.query_params.get(self.query_param)
        if provided and provided == self.expected_api_key:
            return await call_next(request)

        return JSONResponse({"detail": "Unauthorized"}, status_code=401)
