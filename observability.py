from __future__ import annotations
import json
import logging
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)

_metrics_hook: Callable[[str, dict], None] | None = None


class CHPEvent:
    SELECT_CHUNKS_CALLED    = "chp.select_chunks.called"
    CHUNK_SELECTED          = "chp.chunk.selected"
    CHUNK_EXCLUDED          = "chp.chunk.excluded"
    MUST_CARRY_DELIVERED    = "chp.must_carry.delivered"
    MUST_CARRY_MISSED       = "chp.must_carry.missed"
    LEDGER_FALLBACK_TRIGGERED = "chp.ledger_fallback.triggered"
    LEDGER_FALLBACK_RECOVERED = "chp.ledger_fallback.recovered"
    LEDGER_WRITE            = "chp.ledger.write"
    LEDGER_QUERY            = "chp.ledger.query"
    TOKEN_REDUCTION         = "chp.token_reduction"


def set_metrics_hook(fn: Callable[[str, dict], None] | None) -> None:
    """Register a metrics callback: fn(event: str, data: dict) -> None.

    Pass None to deregister. Hook exceptions are swallowed — never crashes CHP.
    """
    global _metrics_hook
    _metrics_hook = fn


def emit(event: str, data: dict) -> None:
    log.info(json.dumps({"chp_event": event, **data}))
    if _metrics_hook is not None:
        try:
            _metrics_hook(event, data)
        except Exception:
            pass


class Timer:
    """Context manager: elapsed_ms attribute after exit."""
    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


class SessionTokenTracker:
    """
    Accumulates TOKEN_REDUCTION events across hops for one session and emits
    a final summary when close() is called.

    Usage:
        tracker = SessionTokenTracker("session-abc")
        chp.set_metrics_hook(tracker.on_event)

        # ... run your pipeline (select_chunks fires TOKEN_REDUCTION per hop) ...

        summary = tracker.close()
        # summary = {
        #   "session_id": "session-abc",
        #   "hops": 5,
        #   "total_tokens_in": 4775,
        #   "total_tokens_out": 1500,
        #   "overall_reduction_pct": 68.6,
        # }

    Thread-safe: safe to use from concurrent agent threads in the same session.

    Chaining with an existing hook:
        tracker = SessionTokenTracker("s1", upstream_hook=my_prometheus_hook)
        chp.set_metrics_hook(tracker.on_event)
    """

    SESSION_SUMMARY = "chp.session.token_summary"

    def __init__(
        self,
        session_id: str,
        upstream_hook: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.session_id = session_id
        self._upstream = upstream_hook
        self._lock = threading.Lock()
        self._hops: int = 0
        self._tokens_in: int = 0
        self._tokens_out: int = 0

    def on_event(self, event: str, data: dict) -> None:
        """Pass as the metrics hook: chp.set_metrics_hook(tracker.on_event)"""
        if event == CHPEvent.TOKEN_REDUCTION:
            sid = data.get("session_id")
            if sid is None or sid == self.session_id:
                with self._lock:
                    self._hops += 1
                    self._tokens_in  += data.get("tokens_in",  0)
                    self._tokens_out += data.get("tokens_out", 0)
        if self._upstream is not None:
            try:
                self._upstream(event, data)
            except Exception:
                pass

    def close(self) -> dict:
        """
        Emit SESSION_SUMMARY event and return the summary dict.
        Call once at end of session (after last select_chunks hop).
        """
        with self._lock:
            hops       = self._hops
            tokens_in  = self._tokens_in
            tokens_out = self._tokens_out

        reduction_pct = round(
            (1 - tokens_out / tokens_in) * 100, 1
        ) if tokens_in > 0 else 0.0
        summary = {
            "session_id":           self.session_id,
            "hops":                 hops,
            "total_tokens_in":      tokens_in,
            "total_tokens_out":     tokens_out,
            "overall_reduction_pct": reduction_pct,
        }
        emit(SessionTokenTracker.SESSION_SUMMARY, summary)
        return summary

    def reset(self) -> None:
        """Reset counters — reuse the tracker for a new session."""
        with self._lock:
            self._hops = 0
            self._tokens_in = 0
            self._tokens_out = 0
