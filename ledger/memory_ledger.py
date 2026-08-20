"""
InMemoryLedger — CHP ledger backed by plain Python dicts.

Zero dependencies. Zero disk I/O. Fastest possible writes and queries.
Lost on process exit — suitable for tests, CI, and ephemeral pipelines.

Usage:
    from chp.ledger.memory_ledger import InMemoryLedger

    ledger = InMemoryLedger()
    # identical API to LanceDBLedger / SQLiteLedger
"""
from __future__ import annotations

import json
import re as _re
import threading
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from chp.schema.rationale_envelope import RationaleEnvelope
from chp.ledger.base import LedgerBackend
from chp.observability import CHPEvent, emit, Timer

if TYPE_CHECKING:
    from chp.engine.embedder import Embedder


def _safe_id(value: str, field: str = "id") -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"CHP: {field} must be a non-empty string")
    if not _re.match(r'^[a-zA-Z0-9_\-.:]+$', value):
        raise ValueError(
            f"CHP: {field}={value!r} contains invalid characters. "
            "Only alphanumeric, hyphen, underscore, dot, and colon are allowed."
        )
    return value


class InMemoryLedger(LedgerBackend):
    """Thread-safe in-memory ledger. No persistence — resets on process restart."""

    def __init__(self) -> None:
        # chunk_id → {"content", "source_agent", "token_cost", "created_at"}
        self._chunks: dict[str, dict] = {}
        # list of ledger row dicts
        self._rows: list[dict] = []
        self._lock = threading.Lock()

    # ── Write ──────────────────────────────────────────────────────────────────

    def write(
        self,
        session_id: str,
        hop_number: int,
        from_agent: str,
        to_agent: str,
        envelope: RationaleEnvelope,
        embedder: "Embedder | None" = None,
    ) -> str:
        session_id = _safe_id(session_id, "session_id")
        from_agent = _safe_id(from_agent, "from_agent")
        to_agent   = _safe_id(to_agent,   "to_agent")
        _safe_id(envelope.chunk_id, "chunk_id")

        with Timer() as t:
            with self._lock:
                # Duplicate check
                for row in self._rows:
                    if (
                        row["session_id"] == session_id
                        and row["chunk_id"] == envelope.chunk_id
                        and row["hop_number"] == hop_number
                    ):
                        raise RuntimeError(
                            f"duplicate entry: session={session_id} "
                            f"chunk={envelope.chunk_id} hop={hop_number}"
                        )

                # Upsert chunk (dedup by chunk_id)
                if envelope.chunk_id not in self._chunks:
                    self._chunks[envelope.chunk_id] = {
                        "content":      envelope.content,
                        "source_agent": envelope.source_agent,
                        "token_cost":   envelope.token_cost,
                        "created_at":   datetime.now(timezone.utc).isoformat(),
                    }

                ledger_id = str(uuid.uuid4())
                self._rows.append({
                    "ledger_id":     ledger_id,
                    "session_id":    session_id,
                    "hop_number":    hop_number,
                    "from_agent":    from_agent,
                    "to_agent":      to_agent,
                    "chunk_id":      envelope.chunk_id,
                    "score":         float(envelope.score),
                    "must_carry":    bool(envelope.must_carry),
                    "hop_sequence":  envelope.hop_sequence,
                    "selected_because": envelope.selected_because,
                    "source_turn":   envelope.source_turn,
                    "timestamp":     datetime.now(timezone.utc).isoformat(),
                })

        emit(CHPEvent.LEDGER_WRITE, {
            "session_id": session_id, "hop_number": hop_number,
            "from_agent": from_agent, "to_agent": to_agent,
            "chunk_id": envelope.chunk_id, "elapsed_ms": round(t.elapsed_ms, 2),
        })
        return ledger_id

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(
        self,
        session_id: str,
        agent_id: str | None = None,
    ) -> list[RationaleEnvelope]:
        session_id = _safe_id(session_id, "session_id")
        with Timer() as t:
            with self._lock:
                rows = [
                    r for r in self._rows
                    if r["session_id"] == session_id
                    and (agent_id is None or r["to_agent"] == agent_id)
                ]
                result = [self._hydrate(r) for r in rows]
        emit(CHPEvent.LEDGER_QUERY, {
            "session_id": session_id, "agent_id": agent_id,
            "rows_returned": len(result), "elapsed_ms": round(t.elapsed_ms, 2),
        })
        return result

    def query_hop(
        self,
        session_id: str,
        hop_number: int,
    ) -> list[RationaleEnvelope]:
        session_id = _safe_id(session_id, "session_id")
        with self._lock:
            rows = [
                r for r in self._rows
                if r["session_id"] == session_id and r["hop_number"] == hop_number
            ]
            return [self._hydrate(r) for r in rows]

    # query_by_meaning → base returns [] (no vector support)

    # ── Maintenance ────────────────────────────────────────────────────────────

    def prune(self, session_id: str) -> int:
        session_id = _safe_id(session_id, "session_id")
        with self._lock:
            before = len(self._rows)
            self._rows = [r for r in self._rows if r["session_id"] != session_id]
            return before - len(self._rows)

    def prune_older_than(self, cutoff_iso: str) -> int:
        with self._lock:
            before = len(self._rows)
            self._rows = [r for r in self._rows if r["timestamp"] >= cutoff_iso]
            return before - len(self._rows)

    def prune_orphan_chunks(self) -> int:
        with self._lock:
            live = {r["chunk_id"] for r in self._rows}
            orphans = [cid for cid in self._chunks if cid not in live]
            for cid in orphans:
                del self._chunks[cid]
            return len(orphans)

    def stats(self) -> dict:
        with self._lock:
            return {
                "ledger_rows": len(self._rows),
                "chunk_rows":  len(self._chunks),
            }

    def clear(self) -> None:
        """Reset all data — useful between tests."""
        with self._lock:
            self._rows.clear()
            self._chunks.clear()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _hydrate(self, row: dict) -> RationaleEnvelope:
        chunk = self._chunks.get(row["chunk_id"], {})
        return RationaleEnvelope(
            chunk_id=row["chunk_id"],
            content=chunk.get("content", ""),
            source_agent=chunk.get("source_agent", row.get("from_agent", "")),
            source_turn=row["source_turn"],
            hop_sequence=row["hop_sequence"],
            selected_because=row["selected_because"],
            score=row["score"],
            must_carry=row["must_carry"],
            token_cost=chunk.get("token_cost", 0),
            ledger_id=row["ledger_id"],
        )
