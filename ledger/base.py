"""
LedgerBackend — abstract contract all CHP ledger implementations must satisfy.

Swap by passing any LedgerBackend to framework adapters and select_chunks():

    from chp.ledger.sqlite_ledger import SQLiteLedger
    ledger = SQLiteLedger("/data/chp.db")
    # use exactly like CHPLedger / LanceDBLedger

Implementations shipped:
    LanceDBLedger  (chp.ledger.lancedb_ledger) — embedded vector DB, default
    SQLiteLedger   (chp.ledger.sqlite_ledger)  — stdlib sqlite3, zero extra deps
    InMemoryLedger (chp.ledger.memory_ledger)  — RAM only, tests / CI
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from chp.schema.rationale_envelope import RationaleEnvelope

if TYPE_CHECKING:
    from chp.engine.embedder import Embedder


class LedgerBackend(ABC):
    """
    Minimal interface all ledger backends must implement.

    Sync methods are the primary interface.
    Async counterparts default to asyncio.to_thread(sync_method) so backends
    only need to override the async versions if they have a native async driver.
    """

    # ── Write ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def write(
        self,
        session_id: str,
        hop_number: int,
        from_agent: str,
        to_agent: str,
        envelope: RationaleEnvelope,
        embedder: "Embedder | None" = None,
    ) -> str:
        """Append one envelope. Returns ledger_id. Raises RuntimeError on duplicate."""

    # ── Query ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def query(
        self,
        session_id: str,
        agent_id: str | None = None,
    ) -> list[RationaleEnvelope]:
        """Return all envelopes for a session, optionally filtered by to_agent."""

    @abstractmethod
    def query_hop(
        self,
        session_id: str,
        hop_number: int,
    ) -> list[RationaleEnvelope]:
        """Return all envelopes written at a specific hop."""

    def query_by_meaning(
        self,
        query_text: str,
        embedder: "Embedder",
        session_id: str | None = None,
        top_k: int = 10,
    ) -> list[RationaleEnvelope]:
        """
        Semantic (ANN vector) search over stored chunks.

        Default implementation: returns empty list.
        Override in backends that support vector search (e.g. LanceDBLedger).
        Backends without vector support (SQLite, InMemory) return [] gracefully.
        """
        return []

    # ── Async API (default: wrap sync in thread pool) ──────────────────────────

    async def awrite(
        self,
        session_id: str,
        hop_number: int,
        from_agent: str,
        to_agent: str,
        envelope: RationaleEnvelope,
        embedder: "Embedder | None" = None,
    ) -> str:
        import asyncio
        return await asyncio.to_thread(
            self.write, session_id, hop_number, from_agent, to_agent, envelope, embedder
        )

    async def aquery(
        self,
        session_id: str,
        agent_id: str | None = None,
    ) -> list[RationaleEnvelope]:
        import asyncio
        return await asyncio.to_thread(self.query, session_id, agent_id)

    async def aquery_hop(
        self,
        session_id: str,
        hop_number: int,
    ) -> list[RationaleEnvelope]:
        import asyncio
        return await asyncio.to_thread(self.query_hop, session_id, hop_number)

    async def aquery_by_meaning(
        self,
        query_text: str,
        embedder: "Embedder",
        session_id: str | None = None,
        top_k: int = 10,
    ) -> list[RationaleEnvelope]:
        import asyncio
        return await asyncio.to_thread(
            self.query_by_meaning, query_text, embedder, session_id, top_k
        )

    # ── Maintenance ────────────────────────────────────────────────────────────

    @abstractmethod
    def prune(self, session_id: str) -> int:
        """Delete all rows for session. Returns rows deleted."""

    def prune_older_than(self, cutoff_iso: str) -> int:
        """Delete rows with timestamp < cutoff_iso. Override if supported."""
        return 0

    def prune_orphan_chunks(self) -> int:
        """Delete unreferenced chunk content. Override if backend deduplicates."""
        return 0

    def compact(self) -> None:
        """Compact storage (merge files, rebuild indexes). No-op by default."""

    async def aprune(self, session_id: str) -> int:
        import asyncio
        return await asyncio.to_thread(self.prune, session_id)

    async def aprune_orphan_chunks(self) -> int:
        import asyncio
        return await asyncio.to_thread(self.prune_orphan_chunks)

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return row counts and any backend-specific metrics."""
        return {}
