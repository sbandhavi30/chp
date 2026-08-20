"""
MongoDBLedger — CHP ledger backed by MongoDB via pymongo.

Install:
    pip install "chp[mongodb]"

This backend stores chunk content and ledger routing records in two MongoDB
collections (``chp_chunks`` and ``chp_ledger``) within a single database.
It is suitable for single-node MongoDB, replica sets, and Atlas clusters.

No vector search is provided (``query_by_meaning`` returns [] — use
LanceDBLedger if ANN is needed). Writes are serialised through a
``threading.Lock`` to guarantee atomic duplicate-check + insert pairs;
reads are lock-free (pymongo connections are thread-safe for reads).

Usage — local MongoDB:
    from chp.ledger.mongodb_ledger import MongoDBLedger

    ledger = MongoDBLedger("mongodb://localhost:27017")
    # or explicit db name:
    ledger = MongoDBLedger("mongodb://localhost:27017", db_name="chp_prod")

Usage — Atlas:
    ledger = MongoDBLedger(
        "mongodb+srv://user:password@cluster0.example.mongodb.net",
        db_name="chp",
    )

Required permissions on the target database:
    - find, insert, update, delete on chp_chunks and chp_ledger
    - createIndex on chp_chunks and chp_ledger
    - dbStats / collStats (for stats())
    - compact (optional, only required for compact(); needs dbAdmin or higher on Atlas)

Thread safety: threading.Lock serialises duplicate-check + write pairs.
Multi-process / multi-node: the unique compound index on
(session_id, chunk_id, hop_number) in chp_ledger acts as the authoritative
duplicate guard at the database level. The in-process lock is an optimisation
that converts silent constraint errors into clear RuntimeErrors before they
reach the driver.
"""
from __future__ import annotations

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
    """Validate that *value* is a non-empty string with safe characters."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"CHP: {field} must be a non-empty string")
    if not _re.match(r'^[a-zA-Z0-9_\-.:]+$', value):
        raise ValueError(
            f"CHP: {field}={value!r} contains invalid characters. "
            "Only alphanumeric, hyphen, underscore, dot, and colon are allowed."
        )
    return value


class MongoDBLedger(LedgerBackend):
    """CHP ledger backend persisted in MongoDB.

    Two collections are used:
    - ``chp_chunks``  — deduplicated chunk content (keyed by chunk_id).
    - ``chp_ledger``  — routing records: who forwarded what to whom, at which hop.
    """

    def __init__(self, uri: str, db_name: str = "chp") -> None:
        """
        Args:
            uri:     MongoDB connection URI.
                     Examples:
                       ``"mongodb://localhost:27017"``
                       ``"mongodb+srv://user:pass@cluster.example.mongodb.net"``
            db_name: Name of the MongoDB database to use (default ``"chp"``).

        Raises:
            ImportError: if pymongo is not installed. Run
                         ``pip install "chp[mongodb]"`` to fix this.
        """
        try:
            from pymongo import MongoClient, ASCENDING
        except ImportError as exc:
            raise ImportError(
                "MongoDBLedger requires pymongo. "
                'Install it with:  pip install "chp[mongodb]"'
            ) from exc

        self._client = MongoClient(uri)
        self._db = self._client[db_name]
        self._chunks: "pymongo.collection.Collection" = self._db["chp_chunks"]
        self._ledger: "pymongo.collection.Collection" = self._db["chp_ledger"]
        self._lock = threading.Lock()

        # ── Indexes ────────────────────────────────────────────────────────────
        # chp_chunks: unique on chunk_id (primary dedup guard)
        self._chunks.create_index(
            [("chunk_id", ASCENDING)],
            unique=True,
            name="uidx_chunks_chunk_id",
        )

        # chp_ledger: query-support indexes
        self._ledger.create_index(
            [("session_id", ASCENDING)],
            name="idx_ledger_session_id",
        )
        self._ledger.create_index(
            [("session_id", ASCENDING), ("to_agent", ASCENDING)],
            name="idx_ledger_session_to_agent",
        )
        self._ledger.create_index(
            [("session_id", ASCENDING), ("hop_number", ASCENDING)],
            name="idx_ledger_session_hop_number",
        )
        # Authoritative duplicate-detection index
        self._ledger.create_index(
            [
                ("session_id", ASCENDING),
                ("chunk_id", ASCENDING),
                ("hop_number", ASCENDING),
            ],
            unique=True,
            name="uidx_ledger_session_chunk_hop",
        )

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
        """Append one envelope to the ledger. Returns the ledger_id string.

        Raises:
            RuntimeError: if an identical (session_id, chunk_id, hop_number)
                          entry already exists.
            ValueError:   if any ID argument fails the safe-character check.
        """
        session_id = _safe_id(session_id, "session_id")
        from_agent = _safe_id(from_agent, "from_agent")
        to_agent   = _safe_id(to_agent,   "to_agent")
        _safe_id(envelope.chunk_id, "chunk_id")

        chunk_id = envelope.chunk_id

        with Timer() as t:
            with self._lock:
                # ── Duplicate check ────────────────────────────────────────────
                existing = self._ledger.find_one({
                    "session_id": session_id,
                    "chunk_id":   chunk_id,
                    "hop_number": hop_number,
                })
                if existing is not None:
                    raise RuntimeError(
                        f"duplicate entry: session={session_id} "
                        f"chunk={chunk_id} hop={hop_number}"
                    )

                # ── Chunk upsert (insert only if chunk_id is new) ──────────────
                self._chunks.update_one(
                    {"chunk_id": chunk_id},
                    {
                        "$setOnInsert": {
                            "chunk_id":     chunk_id,
                            "content":      envelope.content,
                            "source_agent": envelope.source_agent,
                            "token_cost":   int(envelope.token_cost),
                            "created_at":   datetime.now(timezone.utc).isoformat(),
                        }
                    },
                    upsert=True,
                )

                # ── Ledger record ──────────────────────────────────────────────
                ledger_id = str(uuid.uuid4())
                self._ledger.insert_one({
                    "ledger_id":        ledger_id,
                    "session_id":       session_id,
                    "hop_number":       int(hop_number),
                    "from_agent":       from_agent,
                    "to_agent":         to_agent,
                    "chunk_id":         chunk_id,
                    "score":            float(envelope.score),
                    "must_carry":       bool(envelope.must_carry),
                    "hop_sequence":     list(envelope.hop_sequence),
                    "selected_because": list(envelope.selected_because),
                    "source_turn":      int(envelope.source_turn),
                    "timestamp":        datetime.now(timezone.utc).isoformat(),
                })

        emit(CHPEvent.LEDGER_WRITE, {
            "session_id":  session_id,
            "hop_number":  hop_number,
            "from_agent":  from_agent,
            "to_agent":    to_agent,
            "chunk_id":    chunk_id,
            "elapsed_ms":  round(t.elapsed_ms, 2),
        })
        return ledger_id

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(
        self,
        session_id: str,
        agent_id: str | None = None,
    ) -> list[RationaleEnvelope]:
        """Return all envelopes for *session_id*, optionally filtered by to_agent."""
        session_id = _safe_id(session_id, "session_id")

        with Timer() as t:
            filt: dict = {"session_id": session_id}
            if agent_id:
                agent_id = _safe_id(agent_id, "agent_id")
                filt["to_agent"] = agent_id

            ledger_docs = list(self._ledger.find(filt))
            result = []
            for ldoc in ledger_docs:
                chunk_doc = self._chunks.find_one({"chunk_id": ldoc["chunk_id"]})
                if chunk_doc is not None:
                    result.append(self._hydrate(ldoc, chunk_doc))

        emit(CHPEvent.LEDGER_QUERY, {
            "session_id":    session_id,
            "agent_id":      agent_id,
            "rows_returned": len(result),
            "elapsed_ms":    round(t.elapsed_ms, 2),
        })
        return result

    def query_hop(
        self,
        session_id: str,
        hop_number: int,
    ) -> list[RationaleEnvelope]:
        """Return all envelopes written at *hop_number* within *session_id*."""
        session_id = _safe_id(session_id, "session_id")

        ledger_docs = list(self._ledger.find({
            "session_id": session_id,
            "hop_number": int(hop_number),
        }))
        result = []
        for ldoc in ledger_docs:
            chunk_doc = self._chunks.find_one({"chunk_id": ldoc["chunk_id"]})
            if chunk_doc is not None:
                result.append(self._hydrate(ldoc, chunk_doc))
        return result

    # query_by_meaning not supported — base class returns [] automatically

    # ── Maintenance ────────────────────────────────────────────────────────────

    def prune(self, session_id: str) -> int:
        """Delete all ledger rows for *session_id*. Returns the number deleted."""
        session_id = _safe_id(session_id, "session_id")
        with self._lock:
            result = self._ledger.delete_many({"session_id": session_id})
        return result.deleted_count

    def prune_older_than(self, cutoff_iso: str) -> int:
        """Delete ledger rows whose timestamp is lexicographically before *cutoff_iso*.

        Because timestamps are stored as ISO-8601 strings (UTC), lexicographic
        comparison equals chronological comparison for the same timezone offset.

        Args:
            cutoff_iso: ISO-8601 timestamp string, e.g. ``"2025-01-01T00:00:00+00:00"``.

        Returns:
            Number of ledger rows deleted.
        """
        with self._lock:
            result = self._ledger.delete_many({"timestamp": {"$lt": cutoff_iso}})
        return result.deleted_count

    def prune_orphan_chunks(self) -> int:
        """Delete chunk documents that are no longer referenced by any ledger row.

        Collects the full set of chunk_ids currently in chp_ledger, then removes
        any chp_chunks document whose chunk_id is NOT in that set.

        Returns:
            Number of chunk documents deleted.
        """
        with self._lock:
            referenced = self._ledger.distinct("chunk_id")
            referenced_set = set(referenced)
            if referenced_set:
                result = self._chunks.delete_many(
                    {"chunk_id": {"$nin": list(referenced_set)}}
                )
            else:
                # No ledger rows at all — remove every chunk
                result = self._chunks.delete_many({})
        return result.deleted_count

    def compact(self) -> None:
        """Run MongoDB's compact command on both collections.

        This rewrites collection data contiguously and rebuilds indexes,
        reclaiming fragmented space. Requires the ``dbAdmin`` role on the
        database (or equivalent on Atlas dedicated clusters).

        On Atlas shared (M0/M2/M5) tiers this command is not available and
        will raise ``OperationFailure``; the exception is caught and silently
        ignored to avoid breaking maintenance pipelines.
        """
        try:
            self._db.command("compact", "chp_chunks")
        except Exception:
            pass
        try:
            self._db.command("compact", "chp_ledger")
        except Exception:
            pass

    def stats(self) -> dict:
        """Return document counts for both collections.

        Uses ``estimated_document_count()`` which is fast (metadata-based)
        but may be slightly inaccurate on unclean shutdown; accurate enough
        for observability dashboards.

        Returns:
            ``{"ledger_rows": int, "chunk_rows": int}``
        """
        return {
            "ledger_rows": self._ledger.estimated_document_count(),
            "chunk_rows":  self._chunks.estimated_document_count(),
        }

    def close(self) -> None:
        """Close the underlying MongoClient and release its connection pool."""
        self._client.close()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _hydrate(
        self,
        ledger_doc: dict,
        chunk_doc: dict,
    ) -> RationaleEnvelope:
        """Reconstruct a :class:`RationaleEnvelope` from two MongoDB documents.

        Args:
            ledger_doc: A document from the ``chp_ledger`` collection.
            chunk_doc:  The corresponding document from ``chp_chunks``.

        Returns:
            A fully populated :class:`RationaleEnvelope`.
        """
        return RationaleEnvelope(
            chunk_id=chunk_doc["chunk_id"],
            content=chunk_doc["content"],
            source_agent=chunk_doc["source_agent"],
            source_turn=int(ledger_doc.get("source_turn", 0)),
            hop_sequence=list(ledger_doc.get("hop_sequence", [])),
            selected_because=list(ledger_doc.get("selected_because", [])),
            score=float(ledger_doc["score"]),
            must_carry=bool(ledger_doc["must_carry"]),
            token_cost=int(chunk_doc["token_cost"]),
            ledger_id=ledger_doc.get("ledger_id"),
        )
