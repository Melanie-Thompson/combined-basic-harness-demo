"""Tunnel authentication — validates incoming session tunnel connections.

When a harness opens a session tunnel, every request carries:
    X-Session-ID:        the harness's session identifier
    X-Tunnel-ID:         unique tunnel identifier
    X-Request-Timestamp: when the request was sent
    X-Request-Seq:       monotonic sequence number

This module validates those headers and provides middleware for FastAPI.
"""

import logging
import time
from dataclasses import dataclass, field

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger("tunnel_auth")

MAX_CLOCK_SKEW = 60.0
MAX_SESSION_DURATION = 3600.0


@dataclass
class TunnelSession:
    session_id: str
    tunnel_id: str
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    request_count: int = 0
    last_seq: int = -1


class TunnelRegistry:
    """Tracks active tunnel sessions from harnesses."""

    def __init__(self):
        self._sessions: dict[str, TunnelSession] = {}

    def validate(self, session_id: str, tunnel_id: str, timestamp: float, seq: int) -> str | None:
        """Validate a tunnel request. Returns an error message or None if valid."""
        now = time.time()

        if abs(now - timestamp) > MAX_CLOCK_SKEW:
            return f"clock skew too large: {abs(now - timestamp):.1f}s (max {MAX_CLOCK_SKEW}s)"

        key = f"{session_id}:{tunnel_id}"
        session = self._sessions.get(key)

        if session is None:
            self._sessions[key] = TunnelSession(
                session_id=session_id,
                tunnel_id=tunnel_id,
            )
            log.info("new tunnel session: %s (tunnel %s)", session_id, tunnel_id)
            return None

        if now - session.first_seen > MAX_SESSION_DURATION:
            return f"session expired after {now - session.first_seen:.0f}s"

        if seq <= session.last_seq:
            return f"sequence number {seq} <= last seen {session.last_seq} (replay?)"

        session.last_seen = now
        session.request_count += 1
        session.last_seq = seq
        return None

    def close_session(self, session_id: str) -> int:
        """Close all tunnels for a session. Returns count closed."""
        to_remove = [k for k in self._sessions if k.startswith(f"{session_id}:")]
        for k in to_remove:
            del self._sessions[k]
        if to_remove:
            log.info("closed %d tunnels for session %s", len(to_remove), session_id)
        return len(to_remove)

    @property
    def active_sessions(self) -> list[dict]:
        return [
            {
                "session_id": s.session_id,
                "tunnel_id": s.tunnel_id,
                "request_count": s.request_count,
                "age_seconds": time.time() - s.first_seen,
            }
            for s in self._sessions.values()
        ]


_registry = TunnelRegistry()


def get_tunnel_registry() -> TunnelRegistry:
    return _registry


class TunnelAuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that validates tunnel headers on incoming requests.

    Requests without tunnel headers pass through (they're regular API calls).
    Requests with tunnel headers are validated against the tunnel registry.
    """

    async def dispatch(self, request: Request, call_next):
        session_id = request.headers.get("X-Session-ID")
        tunnel_id = request.headers.get("X-Tunnel-ID")

        if not session_id or not tunnel_id:
            return await call_next(request)

        timestamp_str = request.headers.get("X-Request-Timestamp", "")
        seq_str = request.headers.get("X-Request-Seq", "-1")

        try:
            timestamp = float(timestamp_str)
        except (ValueError, TypeError):
            from starlette.responses import JSONResponse
            return JSONResponse(
                {"error": "invalid X-Request-Timestamp"},
                status_code=400,
            )

        try:
            seq = int(seq_str)
        except (ValueError, TypeError):
            from starlette.responses import JSONResponse
            return JSONResponse(
                {"error": "invalid X-Request-Seq"},
                status_code=400,
            )

        error = _registry.validate(session_id, tunnel_id, timestamp, seq)
        if error:
            from starlette.responses import JSONResponse
            log.warning("tunnel auth failed: %s (session=%s, tunnel=%s)",
                        error, session_id, tunnel_id)
            return JSONResponse(
                {"error": f"tunnel auth: {error}"},
                status_code=403,
            )

        request.state.session_id = session_id
        request.state.tunnel_id = tunnel_id
        return await call_next(request)
