"""
PostgresLedger — CHP ledger backed by PostgreSQL via psycopg2.

Install:
    pip install "chp[postgres]"

Usage:
    from chp.ledger.postgres_ledger import PostgresLedger

    ledger = PostgresLedger("postgresql://user:pass@host:5432/chp_db")
    ledger.write(session_id, hop_number, from_agent, to_agent, envelope)
    results = ledger.query(session_id)
    ledger.close()

Thread safety:
    Uses psycopg2.pool.ThreadedConnectionPool — each thread checks out its own
    connection, runs the query, and returns it. Safe for 20+ concurrent agents.
    Pool size defaults to minconn=2, maxconn=20; tune via constructor args.

Multi-node (K8s / distributed):
    Postgres MVCC + ON CONFLICT DO NOTHING make this safe for multi-node
    deployments without any additional lock backend. No RedisLockBackend needed.
    Use PgBouncer in front for 100+ agent connections.

No vector search (query_by_meaning returns []). Use LanceDBLedger if ANN needed.
"""
from __future__ import annotations

import json
import re as _re
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Generator

from chp.schema.rationale_envelope import RationaleEnvelope
from chp.ledger.base import LedgerBackend
from chp.observability import CHPEvent, emit, Timer

if TYPE_CHECKING:
    from chp.engine.embedder import Embedder

try:
    import psycopg2
    import psycopg2.pool
    import psycopg2.extras
    import psycopg2.extensions
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "psycopg2 is required for PostgresLedger. "
        'Install with:  pip install "chp[postgres]"'
    ) from _err


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
    score         FLOAT NOT NULL,
    must_carry    BOOLEAN NOT NULL,
    metadata_json TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    UNIQUE (session_id, chunk_id, hop_number)
);

CREATE INDEX IF NOT EXISTS idx_ledger_session
    ON chp_ledger(session_id);

CREATE INDEX IF NOT EXISTS idx_ledger_session_agent
    ON chp_ledger(session_id, to_agent);

CREATE INDEX IF NOT EXISTS idx_ledger_session_hop
    ON chp_ledger(session_id, hop_number);
"""

_SELECT_COLS = (
    "l.ledger_id, l.session_id, l.hop_number, "
    "l.from_agent, l.to_agent, l.chunk_id, "
    "l.score, l.must_carry, l.metadata_json, l.timestamp, "
    "c.content, c.source_agent, c.token_cost "
    "FROM chp_ledger l JOIN chp_chunks c ON l.chunk_id = c.chunk_id"
)


def _safe_id(value: str, field: str = "id") -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"CHP: {field} must be a non-empty string")
    if not _re.match(r'^[a-zA-Z0-9_\-.:]+$', value):
        raise ValueError(
            f"CHP: {field}={value!r} contains invalid characters. "
            "Only alphanumeric, hyphen, underscore, dot, and colon are allowed."
        )
    return value


class PostgresLedger(LedgerBackend):
    """CHP ledger backend using PostgreSQL via a thread-safe connection pool."""

    def __init__(
        self,
        dsn: str,
        minconn: int = 2,
        maxconn: int = 20,
    ) -> None:
        """
        Args:
            dsn:     psycopg2 connection string, e.g.
                     "postgresql://user:pass@host:5432/dbname"
            minconn: Minimum connections kept alive in the pool. Default 2.
            maxconn: Maximum concurrent connections. Default 20.
                     Tune to (expected_concurrent_agents * 1.5).
                     Use PgBouncer externally for 100+ agents.
        """
        self._dsn = dsn
        self._pool = psycopg2.pool.ThreadedConnectionPool(minconn, maxconn, dsn)
        self._pool_lock = threading.Lock()  # guards pool.getconn/putconn
        self._init_schema()

    # ── Connection management ──────────────────────────────────────────────────

    @contextmanager
    def _conn(self) -> Generator[psycopg2.extensions.connection, None, None]:
        """
        Check out a connection from the pool, yield it, return it on exit.
        Rolls back and returns the connection even if an exception is raised.
        Attempts one reconnect if the connection is broken (e.g. after PG restart).
        """
        with self._pool_lock:
            conn = self._pool.getconn()
        try:
            # Health check — reconnect if connection was severed
            if conn.closed:
                self._pool.putconn(conn, close=True)
                with self._pool_lock:
                    conn = self._pool.getconn()
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            with self._pool_lock:
                self._pool.putconn(conn)

    # ── Schema bootstrap ───────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_DDL)
            # commit happens on _conn() exit

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
            with self._conn() as conn:
                with conn.cursor() as cur:
                    # Explicit duplicate check before hitting the UNIQUE constraint
                    # so we raise RuntimeError (expected) not IntegrityError.
                    cur.execute(
                        "SELECT 1 FROM chp_ledger "
                        "WHERE session_id=%s AND chunk_id=%s AND hop_number=%s",
                        (session_id, envelope.chunk_id, hop_number),
                    )
                    if cur.fetchone():
                        raise RuntimeError(
                            f"duplicate entry: session={session_id} "
                            f"chunk={envelope.chunk_id} hop={hop_number}"
                        )

                    # Chunk upsert — ON CONFLICT DO NOTHING deduplicates across nodes
                    cur.execute(
                        "INSERT INTO chp_chunks "
                        "(chunk_id, content, source_agent, token_cost, created_at) "
                        "VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (chunk_id) DO NOTHING",
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
                    cur.execute(
                        "INSERT INTO chp_ledger "
                        "(ledger_id, session_id, hop_number, from_agent, to_agent, "
                        " chunk_id, score, must_carry, metadata_json, timestamp) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (
                            ledger_id, session_id, hop_number, from_agent, to_agent,
                            envelope.chunk_id, float(envelope.score),
                            bool(envelope.must_carry), meta,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                # commit on _conn() exit

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
            with self._conn() as conn:
                with conn.cursor() as cur:
                    if agent_id:
                        agent_id = _safe_id(agent_id, "agent_id")
                        cur.execute(
                            f"SELECT {_SELECT_COLS} "
                            "WHERE l.session_id=%s AND l.to_agent=%s",
                            (session_id, agent_id),
                        )
                    else:
                        cur.execute(
                            f"SELECT {_SELECT_COLS} WHERE l.session_id=%s",
                            (session_id,),
                        )
                    rows = cur.fetchall()

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
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT {_SELECT_COLS} "
                    "WHERE l.session_id=%s AND l.hop_number=%s",
                    (session_id, hop_number),
                )
                rows = cur.fetchall()
        return [self._hydrate(r) for r in rows]

    # query_by_meaning not supported — base class returns [] automatically

    # ── Maintenance ────────────────────────────────────────────────────────────

    def prune(self, session_id: str) -> int:
        session_id = _safe_id(session_id, "session_id")
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chp_ledger WHERE session_id=%s", (session_id,)
                )
                return cur.rowcount

    def prune_older_than(self, cutoff_iso: str) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chp_ledger WHERE timestamp < %s", (cutoff_iso,)
                )
                return cur.rowcount

    def prune_orphan_chunks(self) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM chp_chunks "
                    "WHERE chunk_id NOT IN (SELECT DISTINCT chunk_id FROM chp_ledger)"
                )
                return cur.rowcount

    def compact(self) -> None:
        """
        VACUUM ANALYZE — reclaims storage and refreshes planner statistics.

        VACUUM cannot run inside a transaction block, so this opens a dedicated
        connection with autocommit=True, independent of the pool.
        """
        try:
            conn = psycopg2.connect(self._dsn)
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("VACUUM ANALYZE chp_ledger")
                cur.execute("VACUUM ANALYZE chp_chunks")
            conn.close()
        except Exception:
            pass

    def stats(self) -> dict:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chp_ledger")
                ledger_rows = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM chp_chunks")
                chunk_rows = cur.fetchone()[0]
        return {"ledger_rows": ledger_rows, "chunk_rows": chunk_rows}

    def close(self) -> None:
        """Close all connections in the pool."""
        self._pool.closeall()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _hydrate(self, row: tuple) -> RationaleEnvelope:
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
