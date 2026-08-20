"""
SQLiteLedger — CHP ledger backed by stdlib sqlite3.

Zero extra dependencies beyond Python itself.
Good for: single-node deployments, edge devices, small-scale production,
          teams that already run Postgres/MySQL and want to evaluate CHP cheaply.

No vector search (query_by_meaning returns []). Use LanceDBLedger if ANN is needed.

Usage:
    from chp.ledger.sqlite_ledger import SQLiteLedger

    ledger = SQLiteLedger("/data/chp.db")          # persistent
    ledger = SQLiteLedger(":memory:")               # in-process, lost on exit
    ledger = SQLiteLedger()                         # temp file, auto-deleted

Thread safety: WAL mode + per-connection threading.Lock.
Multi-process: SQLite WAL handles concurrent readers; writes serialized via lock.
Multi-node (K8s): use LanceDBLedger + RedisLockBackend instead.
"""
from __future__ import annotations

import json
import re as _re
import sqlite3
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from chp.schema.rationale_envelope import RationaleEnvelope
from chp.ledger.base import LedgerBackend
from chp.observability import CHPEvent, emit, Timer

if TYPE_CHECKING:
    from chp.engine.embedder import Embedder

_DDL = """
CREATE TABLE IF NOT EXISTS chp_chunks (
    chunk_id     TEXT PRIMARY KEY,
    content      TEXT NOT NULL,
    source_agent TEXT NOT NULL,
    token_cost   INTEGER NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chp_ledger (
    ledger_id     TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    hop_number    INTEGER NOT NULL,
    from_agent    TEXT NOT NULL,
    to_agent      TEXT NOT NULL,
    chunk_id      TEXT NOT NULL,
    score         REAL NOT NULL,
    must_carry    INTEGER NOT NULL,
    metadata_json TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    UNIQUE(session_id, chunk_id, hop_number)
);

CREATE INDEX IF NOT EXISTS idx_ledger_session ON chp_ledger(session_id);
CREATE INDEX IF NOT EXISTS idx_ledger_session_agent ON chp_ledger(session_id, to_agent);
CREATE INDEX IF NOT EXISTS idx_ledger_session_hop ON chp_ledger(session_id, hop_number);
"""


def _safe_id(value: str, field: str = "id") -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"CHP: {field} must be a non-empty string")
    if not _re.match(r'^[a-zA-Z0-9_\-.:]+$', value):
        raise ValueError(
            f"CHP: {field}={value!r} contains invalid characters. "
            "Only alphanumeric, hyphen, underscore, dot, and colon are allowed."
        )
    return value


class SQLiteLedger(LedgerBackend):
    def __init__(self, db_path: str | None = None) -> None:
        """
        Args:
            db_path: Path to SQLite file.
                     None → temp file (auto-deleted on close).
                     ":memory:" → in-process RAM (use InMemoryLedger for tests instead).
        """
        if db_path is None:
            self._tmpfile = tempfile.NamedTemporaryFile(
                suffix=".db", prefix="chp_ledger_", delete=True
            )
            path = self._tmpfile.name
        else:
            self._tmpfile = None
            path = db_path

        self._path = path
        # WAL mode: readers don't block writers, multiple readers concurrent.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_DDL)
        self._conn.commit()
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
                # Duplicate check (UNIQUE constraint also guards, but give clear error)
                cur = self._conn.execute(
                    "SELECT 1 FROM chp_ledger "
                    "WHERE session_id=? AND chunk_id=? AND hop_number=?",
                    (session_id, envelope.chunk_id, hop_number),
                )
                if cur.fetchone():
                    raise RuntimeError(
                        f"duplicate entry: session={session_id} "
                        f"chunk={envelope.chunk_id} hop={hop_number}"
                    )

                # Upsert chunk content (INSERT OR IGNORE — dedup)
                self._conn.execute(
                    "INSERT OR IGNORE INTO chp_chunks "
                    "(chunk_id, content, source_agent, token_cost, created_at) "
                    "VALUES (?,?,?,?,?)",
                    (
                        envelope.chunk_id,
                        envelope.content,
                        envelope.source_agent,
                        envelope.token_cost,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

                ledger_id = str(uuid.uuid4())
                meta = json.dumps({
                    "hop_sequence":     envelope.hop_sequence,
                    "selected_because": envelope.selected_because,
                    "source_turn":      envelope.source_turn,
                })
                self._conn.execute(
                    "INSERT INTO chp_ledger "
                    "(ledger_id, session_id, hop_number, from_agent, to_agent, "
                    " chunk_id, score, must_carry, metadata_json, timestamp) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        ledger_id, session_id, hop_number, from_agent, to_agent,
                        envelope.chunk_id, float(envelope.score),
                        int(envelope.must_carry), meta,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                self._conn.commit()

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
            if agent_id:
                agent_id = _safe_id(agent_id, "agent_id")
                rows = self._conn.execute(
                    "SELECT l.*, c.content, c.source_agent, c.token_cost "
                    "FROM chp_ledger l JOIN chp_chunks c ON l.chunk_id = c.chunk_id "
                    "WHERE l.session_id=? AND l.to_agent=?",
                    (session_id, agent_id),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT l.*, c.content, c.source_agent, c.token_cost "
                    "FROM chp_ledger l JOIN chp_chunks c ON l.chunk_id = c.chunk_id "
                    "WHERE l.session_id=?",
                    (session_id,),
                ).fetchall()
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
        rows = self._conn.execute(
            "SELECT l.*, c.content, c.source_agent, c.token_cost "
            "FROM chp_ledger l JOIN chp_chunks c ON l.chunk_id = c.chunk_id "
            "WHERE l.session_id=? AND l.hop_number=?",
            (session_id, hop_number),
        ).fetchall()
        return [self._hydrate(r) for r in rows]

    # query_by_meaning not supported — base returns [] automatically

    # ── Maintenance ────────────────────────────────────────────────────────────

    def prune(self, session_id: str) -> int:
        session_id = _safe_id(session_id, "session_id")
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM chp_ledger WHERE session_id=?", (session_id,)
            )
            self._conn.commit()
        return cur.rowcount

    def prune_older_than(self, cutoff_iso: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM chp_ledger WHERE timestamp < ?", (cutoff_iso,)
            )
            self._conn.commit()
        return cur.rowcount

    def prune_orphan_chunks(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM chp_chunks WHERE chunk_id NOT IN "
                "(SELECT DISTINCT chunk_id FROM chp_ledger)"
            )
            self._conn.commit()
        return cur.rowcount

    def compact(self) -> None:
        with self._lock:
            self._conn.execute("VACUUM")
            self._conn.commit()

    def stats(self) -> dict:
        ledger_rows = self._conn.execute(
            "SELECT COUNT(*) FROM chp_ledger"
        ).fetchone()[0]
        chunk_rows = self._conn.execute(
            "SELECT COUNT(*) FROM chp_chunks"
        ).fetchone()[0]
        return {"ledger_rows": ledger_rows, "chunk_rows": chunk_rows}

    def close(self) -> None:
        self._conn.close()
        if self._tmpfile is not None:
            self._tmpfile.close()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _hydrate(self, row: tuple) -> RationaleEnvelope:
        # Row order: ledger columns + content, source_agent, token_cost from JOIN
        # ledger_id, session_id, hop_number, from_agent, to_agent,
        # chunk_id, score, must_carry, metadata_json, timestamp,
        # content, source_agent(chunk), token_cost
        (
            ledger_id, session_id, hop_number, from_agent, to_agent,
            chunk_id, score, must_carry, metadata_json, timestamp,
            content, source_agent, token_cost,
        ) = row
        meta = json.loads(metadata_json)
        return RationaleEnvelope(
            chunk_id=chunk_id,
            content=content,
            source_agent=source_agent,
            source_turn=meta.get("source_turn", 0),
            hop_sequence=meta.get("hop_sequence", []),
            selected_because=meta.get("selected_because", []),
            score=float(score),
            must_carry=bool(must_carry),
            token_cost=token_cost,
            ledger_id=ledger_id,
        )
