from __future__ import annotations
import json
import logging
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
