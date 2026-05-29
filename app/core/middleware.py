"""Request tracking middleware.

Assigns a request_id to every inbound request, binds it to the structlog
context, records latency, and emits an access log line in JSON.
"""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger, request_id_var
from app.core.metrics import HTTP_LATENCY, HTTP_REQUESTS

log = get_logger("http")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or f"REQ_{uuid.uuid4().hex[:12]}"
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        status_code = 500
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            HTTP_REQUESTS.labels(
                method=request.method, path=path, status=str(status_code)
            ).inc()
            HTTP_LATENCY.labels(method=request.method, path=path).observe(elapsed_ms / 1000.0)
            log.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status=status_code,
                latency_ms=round(elapsed_ms, 2),
            )
            request_id_var.reset(token)
