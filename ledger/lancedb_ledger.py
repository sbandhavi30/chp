"""
CHPLedger — production-grade append-only context ledger backed by LanceDB.

Two-table design:
  chp_chunks  — content store, one row per unique chunk_id, holds embedding vector
  chp_ledger  — provenance index, one row per (session, chunk, hop, agent pair)

Benefits:
  - Content deduplication: chunk selected by 10 agents → stored once, not 10×
  - Semantic retrieval: query_by_meaning() uses ANN index on chunk embeddings
  - Scalar index: session_id index on chp_ledger for fast query() at scale
  - TTL: prune(session_id) removes all ledger rows for a closed session
  - Compaction: compact() calls LanceDB optimize() + cleanup_old_versions()
  - Pluggable lock: FileLockBackend (single-node) or RedisLockBackend (K8s/EFS)
  - Rate limiting: token-bucket per agent_id prevents runaway agents flooding disk
"""
from __future__ import annotations

import asyncio
import json
import re as _re
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import lancedb
import numpy as np
import pyarrow as pa

from chp.schema.rationale_envelope import RationaleEnvelope
from chp.observability import CHPEvent, emit, Timer
from chp.ledger.locks import LockBackend, FileLockBackend, NoOpLockBackend
from chp.ledger.base import LedgerBackend

if TYPE_CHECKING:
    from chp.engine.embedder import Embedder

# ── Table schemas ─────────────────────────────────────────────────────────────

_EMBEDDING_DIM = 384  # matches StubEmbedder + sentence-transformers all-MiniLM-L6-v2

_CHUNK_SCHEMA = pa.schema([
    pa.field("chunk_id",     pa.string()),
    pa.field("content",      pa.string()),
    pa.field("source_agent", pa.string()),
    pa.field("token_cost",   pa.int32()),
    pa.field("embedding",    pa.list_(pa.float32(), _EMBEDDING_DIM)),
    pa.field("created_at",   pa.string()),
])

_LEDGER_SCHEMA = pa.schema([
    pa.field("ledger_id",   pa.string()),
    pa.field("session_id",  pa.string()),
    pa.field("hop_number",  pa.int32()),
    pa.field("from_agent",  pa.string()),
    pa.field("to_agent",    pa.string()),
    pa.field("chunk_id",    pa.string()),
    pa.field("score",       pa.float32()),
    pa.field("must_carry",  pa.bool_()),
    pa.field("metadata_json", pa.string()),   # hop_sequence, selected_because, etc.
    pa.field("timestamp",   pa.string()),
])

_CHUNKS_TABLE  = "chp_chunks"
_LEDGER_TABLE  = "chp_ledger"

_LOCK_KEY = "chp_write"


# ── Token-bucket rate limiter ─────────────────────────────────────────────────

class WriteLimiter:
    """
    Per-agent token-bucket rate limiter (zero dependencies).

    Each agent gets an independent bucket refilled at `rate` writes/second
    with a maximum burst of `burst` writes.

    Args:
        rate:   Sustained write rate per agent (writes/second). Default 100.
        burst:  Maximum burst size per agent.  Default 200.
                Set burst=rate for a strict rate with no burst allowance.

    Raises:
        RuntimeError: when an agent exceeds its burst capacity.

    Example — tighten limits for a known chatty agent:
        limiter = WriteLimiter(rate=50, burst=100)
        ledger = CHPLedger(write_limiter=limiter)
    """

    def __init__(self, rate: float = 100.0, burst: int = 200) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if burst <= 0:
            raise ValueError("burst must be > 0")
        self._rate  = rate
        self._burst = float(burst)
        # {agent_id: (tokens_float, last_refill_time)}
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def check(self, agent_id: str) -> None:
        """
        Consume one token for agent_id.

        Raises RuntimeError if the bucket is empty (agent is over rate limit).
        Thread-safe; O(1).
        """
        now = time.monotonic()  # noqa: F821 — `time` imported at top of module
        with self._lock:
            tokens, last = self._buckets.get(agent_id, (self._burst, now))
            # Refill tokens accrued since last check
            elapsed = now - last
            tokens = min(self._burst, tokens + elapsed * self._rate)
            if tokens < 1.0:
                raise RuntimeError(
                    f"CHP rate limit: agent '{agent_id}' exceeded {self._rate:.0f} writes/s "
                    f"(burst={int(self._burst)}). Slow down or raise WriteLimiter limits."
                )
            self._buckets[agent_id] = (tokens - 1.0, now)

    def stats(self) -> dict[str, float]:
        """Return current token levels per agent (snapshot, for observability)."""
        with self._lock:
            return {aid: tokens for aid, (tokens, _) in self._buckets.items()}


def _safe_id(value: str, field: str = "id") -> str:
    """Reject IDs containing characters that could inject into LanceDB WHERE clauses."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"CHP: {field} must be a non-empty string")
    if not _re.match(r'^[a-zA-Z0-9_\-.:]+$', value):
        raise ValueError(
            f"CHP: {field}={value!r} contains invalid characters. "
            "Only alphanumeric, hyphen, underscore, dot, and colon are allowed."
        )
    return value


class LanceDBLedger(LedgerBackend):
    def __init__(
        self,
        db_path: str | None = None,
        lock_backend: LockBackend | None = None,
        write_limiter: WriteLimiter | None = None,
    ) -> None:
        """
        Args:
            db_path:       LanceDB directory. Defaults to a temp dir (good for tests).
            lock_backend:  Concurrency strategy.
                           - None (default): FileLockBackend on db_path (single-node safe).
                           - RedisLockBackend: multi-node / Kubernetes / NFS / EFS.
                           - NoOpLockBackend: testing only.
            write_limiter: Token-bucket rate limiter per agent_id.
                           - None (default): no rate limiting.
                           - WriteLimiter(rate=100, burst=200): 100 writes/s sustained,
                             200 burst, per agent. Raises RuntimeError when exceeded.

        Multi-node example:
            import redis
            from chp.ledger.locks import RedisLockBackend
            r = redis.Redis.from_url("redis://redis-svc:6379/0")
            ledger = CHPLedger("/mnt/efs/chp", lock_backend=RedisLockBackend(r))

        Rate-limiting example:
            from chp.ledger.lancedb_ledger import WriteLimiter
            ledger = CHPLedger(write_limiter=WriteLimiter(rate=50, burst=100))
        """
        path = db_path or tempfile.mkdtemp(prefix="chp_ledger_")
        self._db = lancedb.connect(path)

        existing = self._db.list_tables()
        if _CHUNKS_TABLE not in existing:
            self._db.create_table(_CHUNKS_TABLE, schema=_CHUNK_SCHEMA)
        if _LEDGER_TABLE not in existing:
            self._db.create_table(_LEDGER_TABLE, schema=_LEDGER_SCHEMA)

        self._chunks = self._db.open_table(_CHUNKS_TABLE)
        self._ledger = self._db.open_table(_LEDGER_TABLE)

        # In-process thread safety (always present)
        self._thread_lock = threading.Lock()

        # Cross-process / cross-node lock backend
        if lock_backend is not None:
            self._lock_backend: LockBackend = lock_backend
        else:
            try:
                self._lock_backend = FileLockBackend(path, timeout_s=-1)
            except ImportError:
                # filelock not installed — fall back to no-op (single-thread only)
                self._lock_backend = NoOpLockBackend()

        self._write_limiter: WriteLimiter | None = write_limiter

        # Create scalar index on session_id for fast query() at scale.
        # Silently skipped if LanceDB version doesn't support create_scalar_index.
        self._ensure_scalar_index()

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
        """
        Append one envelope to the ledger.

        Raises RuntimeError on duplicate (session_id, chunk_id, hop_number).
        Content is stored once in chp_chunks; subsequent writes for the same
        chunk_id are no-ops on the content table.
        """
        session_id  = _safe_id(session_id,  "session_id")
        from_agent  = _safe_id(from_agent,  "from_agent")
        to_agent    = _safe_id(to_agent,    "to_agent")
        # Rate check before acquiring lock — fail fast, no contention wasted.
        if self._write_limiter is not None:
            self._write_limiter.check(from_agent)
        with Timer() as t:
            with self._lock_backend.acquire(_LOCK_KEY):   # cross-node / cross-process
                with self._thread_lock:                   # in-process thread safety
                    self._check_duplicate(session_id, envelope.chunk_id, hop_number)
                    self._upsert_chunk(envelope, embedder)

                    ledger_id = str(uuid.uuid4())
                    meta = {
                        "hop_sequence":     envelope.hop_sequence,
                        "selected_because": envelope.selected_because,
                        "source_turn":      envelope.source_turn,
                    }
                    self._ledger.add([{
                        "ledger_id":    ledger_id,
                        "session_id":   session_id,
                        "hop_number":   hop_number,
                        "from_agent":   from_agent,
                        "to_agent":     to_agent,
                        "chunk_id":     envelope.chunk_id,
                        "score":        float(envelope.score),
                        "must_carry":   envelope.must_carry,
                        "metadata_json": json.dumps(meta),
                        "timestamp":    datetime.now(timezone.utc).isoformat(),
                    }])
        emit(CHPEvent.LEDGER_WRITE, {
            "session_id": session_id, "hop_number": hop_number,
            "from_agent": from_agent, "to_agent": to_agent,
            "chunk_id": envelope.chunk_id, "elapsed_ms": round(t.elapsed_ms, 2),
        })
        return ledger_id

    # ── Query — relational ─────────────────────────────────────────────────────

    def query(
        self,
        session_id: str,
        agent_id: str | None = None,
    ) -> list[RationaleEnvelope]:
        """Return all envelopes for a session, optionally filtered to one agent."""
        session_id = _safe_id(session_id, "session_id")
        where = f"session_id = '{session_id}'"
        if agent_id:
            where += f" AND to_agent = '{agent_id}'"
        with Timer() as t:
            rows = self._ledger.search().where(where, prefilter=True).to_list()
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
        """Return all envelopes written at a specific hop."""
        session_id = _safe_id(session_id, "session_id")
        rows = (
            self._ledger.search()
            .where(
                f"session_id = '{session_id}' AND hop_number = {hop_number}",
                prefilter=True,
            )
            .to_list()
        )
        return [self._hydrate(r) for r in rows]

    # ── Query — semantic ───────────────────────────────────────────────────────

    def query_by_meaning(
        self,
        query_text: str,
        embedder: "Embedder",
        session_id: str | None = None,
        top_k: int = 10,
    ) -> list[RationaleEnvelope]:
        """
        ANN vector search over stored chunk embeddings.

        Returns the top_k most semantically similar envelopes.
        Optionally scoped to a single session_id via post-filter.

        Requires embedder — must be the same model used during write().
        Falls back gracefully if no embeddings were stored (stub zeros).
        """
        if session_id is not None:
            session_id = _safe_id(session_id, "session_id")
        query_vec = embedder.embed([query_text])[0].tolist()

        chunk_rows = (
            self._chunks.search(query_vec)
            .limit(top_k * 3)          # over-fetch; we'll join + filter below
            .to_list()
        )
        if not chunk_rows:
            return []

        matched_chunk_ids = {r["chunk_id"] for r in chunk_rows}

        # Join against ledger to get provenance
        results: list[RationaleEnvelope] = []
        for chunk_id in matched_chunk_ids:
            where = f"chunk_id = '{chunk_id}'"
            if session_id:
                where += f" AND session_id = '{session_id}'"
            ledger_rows = (
                self._ledger.search()
                .where(where, prefilter=True)
                .to_list()
            )
            for row in ledger_rows[:1]:  # one envelope per chunk per session is enough
                results.append(self._hydrate(row))
            if len(results) >= top_k:
                break

        return results[:top_k]

    # ── Async API ─────────────────────────────────────────────────────────────
    # All async methods delegate to their sync counterparts via asyncio.to_thread.
    # This is correct for LanceDB (sync I/O library) — offloads blocking I/O to
    # a thread pool without blocking the event loop.

    async def awrite(
        self,
        session_id: str,
        hop_number: int,
        from_agent: str,
        to_agent: str,
        envelope: RationaleEnvelope,
        embedder: "Embedder | None" = None,
    ) -> str:
        return await asyncio.to_thread(
            self.write, session_id, hop_number, from_agent, to_agent, envelope, embedder
        )

    async def aquery(
        self,
        session_id: str,
        agent_id: str | None = None,
    ) -> list[RationaleEnvelope]:
        return await asyncio.to_thread(self.query, session_id, agent_id)

    async def aquery_hop(
        self,
        session_id: str,
        hop_number: int,
    ) -> list[RationaleEnvelope]:
        return await asyncio.to_thread(self.query_hop, session_id, hop_number)

    async def aquery_by_meaning(
        self,
        query_text: str,
        embedder: "Embedder",
        session_id: str | None = None,
        top_k: int = 10,
    ) -> list[RationaleEnvelope]:
        return await asyncio.to_thread(
            self.query_by_meaning, query_text, embedder, session_id, top_k
        )

    async def aprune(self, session_id: str) -> int:
        return await asyncio.to_thread(self.prune, session_id)

    async def aprune_orphan_chunks(self) -> int:
        return await asyncio.to_thread(self.prune_orphan_chunks)

    # ── TTL / maintenance ──────────────────────────────────────────────────────

    def prune(self, session_id: str) -> int:
        """
        Delete all ledger rows for a closed session.

        Does NOT delete from chp_chunks — chunks may be shared across sessions.
        Returns number of rows deleted.
        """
        session_id = _safe_id(session_id, "session_id")
        before = self._ledger.count_rows()
        self._ledger.delete(f"session_id = '{session_id}'")
        after = self._ledger.count_rows()
        return before - after

    def prune_older_than(self, cutoff_iso: str) -> int:
        """
        Delete ledger rows with timestamp < cutoff_iso (ISO 8601 string).

        Example: ledger.prune_older_than("2026-08-01T00:00:00+00:00")
        """
        before = self._ledger.count_rows()
        self._ledger.delete(f"timestamp < '{cutoff_iso}'")
        after = self._ledger.count_rows()
        return before - after

    def prune_orphan_chunks(self) -> int:
        """
        Delete chunk content rows that are no longer referenced by any ledger row.

        O(n) set-difference: one scan of each table, one bulk delete.
        Run after prune() to reclaim disk space from the content store.
        Returns number of chunk rows deleted.
        """
        live_ids: set[str] = {
            r["chunk_id"]
            for r in self._ledger.search().select(["chunk_id"]).to_list()
        }
        all_chunk_ids: list[str] = [
            r["chunk_id"]
            for r in self._chunks.search().select(["chunk_id"]).to_list()
        ]
        orphans = [cid for cid in all_chunk_ids if cid not in live_ids]
        if not orphans:
            return 0
        # Single bulk delete — one LanceDB call instead of N
        quoted = ", ".join(f"'{cid}'" for cid in orphans)
        self._chunks.delete(f"chunk_id IN ({quoted})")
        return len(orphans)

    def compact(self) -> None:
        """
        Compact both LanceDB tables: merge delta files, rebuild vector index,
        remove old versions.  Call periodically (e.g. after each session closes
        or on a background cron).

        Requires `pylance` for full compaction (`pip install pylance`).
        Falls back silently when pylance is absent — no data is lost.
        """
        from datetime import timedelta
        for table in (self._chunks, self._ledger):
            try:
                table.optimize(cleanup_older_than=timedelta(days=0))
            except (ImportError, Exception):
                # pylance not installed or optimize unsupported — skip silently
                pass

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return row counts and approximate disk usage for both tables."""
        return {
            "ledger_rows":  self._ledger.count_rows(),
            "chunk_rows":   self._chunks.count_rows(),
        }

    # ── Indexing ───────────────────────────────────────────────────────────────

    def _ensure_scalar_index(self) -> None:
        """
        Create scalar (bitmap) index on chp_ledger.session_id if not present.

        Turns query() from a full table scan into an index lookup — critical
        past ~50K rows. Silently skipped on LanceDB versions that don't yet
        expose create_scalar_index (pre-0.8).
        """
        try:
            self._ledger.create_scalar_index("session_id", replace=False)
        except Exception:
            # Index already exists, or LanceDB version doesn't support it — fine.
            pass

    def create_index(self, replace: bool = False) -> None:
        """
        (Re)build all indexes:
          - chp_ledger: scalar index on session_id
          - chp_chunks: ANN vector index (IVF_PQ)

        Call after bulk imports or after compact() to ensure indexes are current.

        Args:
            replace: If True, drop and rebuild existing indexes.
        """
        try:
            self._ledger.create_scalar_index("session_id", replace=replace)
        except Exception:
            pass
        try:
            # IVF_PQ is the standard ANN index for LanceDB.
            # num_partitions and num_sub_vectors tuned for 384-dim MiniLM embeddings.
            self._chunks.create_index(
                metric="cosine",
                num_partitions=256,
                num_sub_vectors=96,
                replace=replace,
            )
        except Exception:
            pass

    # ── Internal ───────────────────────────────────────────────────────────────

    def _check_duplicate(self, session_id: str, chunk_id: str, hop_number: int) -> None:
        chunk_id   = _safe_id(chunk_id,   "chunk_id")
        existing = (
            self._ledger.search()
            .where(
                f"session_id = '{session_id}' "
                f"AND chunk_id = '{chunk_id}' "
                f"AND hop_number = {hop_number}",
                prefilter=True,
            )
            .to_list()
        )
        if existing:
            raise RuntimeError(
                f"duplicate entry: session={session_id} chunk={chunk_id} hop={hop_number}"
            )

    def _upsert_chunk(self, envelope: RationaleEnvelope, embedder: "Embedder | None") -> None:
        """Write chunk content + embedding once; skip if chunk_id already present."""
        _safe_id(envelope.chunk_id, "chunk_id")
        existing = (
            self._chunks.search()
            .where(f"chunk_id = '{envelope.chunk_id}'", prefilter=True)
            .to_list()
        )
        if existing:
            return

        if embedder is not None:
            vec = embedder.embed([envelope.content])[0].tolist()
        else:
            vec = [0.0] * _EMBEDDING_DIM

        self._chunks.add([{
            "chunk_id":     envelope.chunk_id,
            "content":      envelope.content,
            "source_agent": envelope.source_agent,
            "token_cost":   envelope.token_cost,
            "embedding":    vec,
            "created_at":   datetime.now(timezone.utc).isoformat(),
        }])

    def _hydrate(self, ledger_row: dict) -> RationaleEnvelope:
        """Reconstruct a RationaleEnvelope by joining ledger row with chunk content."""
        _safe_id(ledger_row["chunk_id"], "chunk_id")
        chunk_rows = (
            self._chunks.search()
            .where(f"chunk_id = '{ledger_row['chunk_id']}'", prefilter=True)
            .to_list()
        )
        content      = chunk_rows[0]["content"]      if chunk_rows else ""
        source_agent = chunk_rows[0]["source_agent"] if chunk_rows else ledger_row.get("from_agent", "")
        token_cost   = chunk_rows[0]["token_cost"]   if chunk_rows else 0

        meta = json.loads(ledger_row["metadata_json"])
        return RationaleEnvelope(
            chunk_id=ledger_row["chunk_id"],
            content=content,
            source_agent=source_agent,
            source_turn=meta.get("source_turn", 0),
            hop_sequence=meta.get("hop_sequence", []),
            selected_because=meta.get("selected_because", []),
            score=float(ledger_row["score"]),
            must_carry=bool(ledger_row["must_carry"]),
            token_cost=token_cost,
            ledger_id=ledger_row["ledger_id"],
        )


# Backward-compatible alias — existing code using CHPLedger keeps working.
CHPLedger = LanceDBLedger
