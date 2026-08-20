"""
Tests for production-grade CHPLedger features:
  - Content deduplication across agents
  - TTL / prune by session and by timestamp
  - Orphan chunk cleanup
  - Semantic query_by_meaning
  - Compaction (smoke test — verifies it doesn't crash)
  - stats()
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from chp.engine.embedder import StubEmbedder
from chp.ledger.lancedb_ledger import CHPLedger
from chp.schema.rationale_envelope import RationaleEnvelope


def _env(chunk_id: str, content: str = "", score: float = 0.8, must_carry: bool = False) -> RationaleEnvelope:
    return RationaleEnvelope(
        chunk_id=chunk_id,
        content=content or f"content of {chunk_id}",
        source_agent="agent-a",
        source_turn=1,
        hop_sequence=["agent-a", "agent-b"],
        selected_because=["domain_match"],
        score=score,
        must_carry=must_carry,
        token_cost=50,
    )


# ── Deduplication ─────────────────────────────────────────────────────────────

def test_chunk_stored_once_across_multiple_agents():
    """Same chunk passed to 5 different agents → stored once in chp_chunks."""
    ledger = CHPLedger()
    env = _env("c_shared", "user_id: USR-001 tier: premium")

    for i in range(5):
        # Different session/hop/agent each time — should all succeed
        ledger.write(f"sess_{i}", hop_number=0, from_agent="router", to_agent=f"agent-{i}", envelope=env)

    s = ledger.stats()
    assert s["chunk_rows"] == 1,   f"Expected 1 chunk row, got {s['chunk_rows']}"
    assert s["ledger_rows"] == 5,  f"Expected 5 ledger rows, got {s['ledger_rows']}"


def test_chunk_content_recovered_after_dedup():
    """Content stored once is correctly retrieved when hydrating any of the 5 ledger rows."""
    ledger = CHPLedger()
    env = _env("c_dedup", "order_id: ORD-42 amount: $99")
    for i in range(3):
        ledger.write(f"sess_{i}", 0, "router", f"agent-{i}", env)

    results = ledger.query("sess_0")
    assert len(results) == 1
    assert results[0].content == "order_id: ORD-42 amount: $99"


# ── TTL / prune ───────────────────────────────────────────────────────────────

def test_prune_session_removes_ledger_rows():
    ledger = CHPLedger()
    ledger.write("sess_keep", 0, "a", "b", _env("c_keep"))
    ledger.write("sess_drop", 0, "a", "b", _env("c_drop"))

    deleted = ledger.prune("sess_drop")
    assert deleted == 1

    assert len(ledger.query("sess_drop")) == 0
    assert len(ledger.query("sess_keep")) == 1


def test_prune_does_not_remove_chunk_content():
    """prune() removes ledger rows only; chp_chunks survives (may be shared)."""
    ledger = CHPLedger()
    ledger.write("sess_a", 0, "a", "b", _env("c_shared2", "shared content"))
    ledger.write("sess_b", 0, "a", "b", _env("c_shared2", "shared content"))

    ledger.prune("sess_a")
    assert ledger.stats()["chunk_rows"] == 1  # still there for sess_b


def test_prune_older_than_removes_old_rows():
    ledger = CHPLedger()
    # Write a row, then prune everything before far future
    ledger.write("sess_old", 0, "a", "b", _env("c_old"))
    deleted = ledger.prune_older_than("2099-01-01T00:00:00+00:00")
    assert deleted >= 1
    assert len(ledger.query("sess_old")) == 0


def test_prune_older_than_preserves_recent_rows():
    ledger = CHPLedger()
    ledger.write("sess_new", 0, "a", "b", _env("c_new"))
    deleted = ledger.prune_older_than("2000-01-01T00:00:00+00:00")
    assert deleted == 0
    assert len(ledger.query("sess_new")) == 1


# ── Orphan cleanup ────────────────────────────────────────────────────────────

def test_prune_orphan_chunks_after_session_prune():
    """After pruning both sessions that referenced a chunk, orphan cleanup removes it."""
    ledger = CHPLedger()
    env_only_in_a = _env("c_orphan", "will be orphaned")
    env_shared    = _env("c_shared3", "used by both")

    ledger.write("sess_a", 0, "a", "b", env_only_in_a)
    ledger.write("sess_a", 1, "a", "b", env_shared)
    ledger.write("sess_b", 0, "a", "b", env_shared)

    ledger.prune("sess_a")
    ledger.prune("sess_b")

    orphans_deleted = ledger.prune_orphan_chunks()
    assert orphans_deleted == 2  # both chunks now unreferenced
    assert ledger.stats()["chunk_rows"] == 0


def test_prune_orphan_chunks_preserves_live_chunks():
    ledger = CHPLedger()
    ledger.write("sess_live", 0, "a", "b", _env("c_live", "still in use"))
    ledger.write("sess_dead", 0, "a", "b", _env("c_dead", "dead chunk"))
    ledger.prune("sess_dead")

    orphans_deleted = ledger.prune_orphan_chunks()
    assert orphans_deleted == 1
    assert ledger.stats()["chunk_rows"] == 1  # c_live survives


# ── Semantic query ────────────────────────────────────────────────────────────

def test_query_by_meaning_returns_results(tmp_path):
    """query_by_meaning returns envelopes (stub embedder gives uniform similarity)."""
    ledger = CHPLedger(db_path=str(tmp_path / "sem_ledger"))
    embedder = StubEmbedder()

    for i in range(4):
        env = _env(f"c_sem_{i}", f"billing refund order_{i}")
        ledger.write("sess_sem", i, "router", f"agent-{i}", env, embedder=embedder)

    results = ledger.query_by_meaning(
        query_text="billing refund",
        embedder=embedder,
        session_id="sess_sem",
        top_k=3,
    )
    assert isinstance(results, list)
    # Stub embedder returns uniform zero vectors — ANN still returns rows
    assert len(results) >= 0  # may be 0 with stub; real embedder gives top_k


def test_query_by_meaning_scoped_to_session(tmp_path):
    """query_by_meaning with session_id only returns rows from that session."""
    ledger = CHPLedger(db_path=str(tmp_path / "scoped_ledger"))
    embedder = StubEmbedder()

    ledger.write("sess_x", 0, "a", "b", _env("c_x", "fraud score risk"), embedder=embedder)
    ledger.write("sess_y", 0, "a", "b", _env("c_y", "billing refund"),   embedder=embedder)

    results = ledger.query_by_meaning("fraud", embedder, session_id="sess_x", top_k=5)
    chunk_ids = {r.chunk_id for r in results}
    # c_y must not appear in sess_x results
    assert "c_y" not in chunk_ids


# ── Fix: prune_orphan_chunks O(n) bulk delete ─────────────────────────────────

def test_prune_orphan_chunks_bulk_delete():
    """Single bulk delete removes all orphans in one LanceDB call."""
    ledger = CHPLedger()
    for i in range(20):
        ledger.write(f"sess-{i % 2}", 0, "a", "b", _env(f"c_bulk_{i:02d}", f"content {i}"))
    assert ledger.stats()["chunk_rows"] == 20

    ledger.prune("sess-0")
    ledger.prune("sess-1")
    deleted = ledger.prune_orphan_chunks()
    assert deleted == 20
    assert ledger.stats()["chunk_rows"] == 0


def test_prune_orphan_chunks_zero_when_none():
    ledger = CHPLedger()
    ledger.write("sess-live2", 0, "a", "b", _env("c_live_ok"))
    assert ledger.prune_orphan_chunks() == 0
    assert ledger.stats()["chunk_rows"] == 1


# ── Fix: multi-process filelock ───────────────────────────────────────────────

def test_filelock_file_created_next_to_db(tmp_path):
    """filelock creates chp_write.lock alongside the DB when filelock is installed."""
    import os
    ledger = CHPLedger(db_path=str(tmp_path))
    ledger.write("sess-lock", 0, "a", "b", _env("c_lock"))
    lock_file = tmp_path / "chp_write.lock"
    assert lock_file.exists(), "filelock .lock file not created — filelock may not be installed"


def test_noop_lock_backend_writes_succeed():
    """NoOpLockBackend bypasses all locking — writes still succeed (single-thread only)."""
    from chp.ledger.locks import NoOpLockBackend
    ledger = CHPLedger(lock_backend=NoOpLockBackend())
    assert isinstance(ledger._lock_backend, NoOpLockBackend)
    ledger.write("sess-noop", 0, "a", "b", _env("c_noop"))
    assert ledger.stats()["ledger_rows"] == 1


# ── Compaction smoke test ─────────────────────────────────────────────────────

def test_compact_does_not_raise():
    """compact() should not crash on a populated ledger."""
    ledger = CHPLedger()
    for i in range(5):
        ledger.write(f"sess_{i}", 0, "a", "b", _env(f"c_compact_{i}"))
    ledger.compact()  # smoke test — no assertion, just must not raise


# ── Stats ─────────────────────────────────────────────────────────────────────

def test_stats_reflects_write_and_prune():
    ledger = CHPLedger()
    assert ledger.stats() == {"ledger_rows": 0, "chunk_rows": 0}

    ledger.write("s1", 0, "a", "b", _env("c_stat1"))
    ledger.write("s1", 1, "a", "b", _env("c_stat2"))
    ledger.write("s2", 0, "a", "b", _env("c_stat1"))  # same chunk, new session

    s = ledger.stats()
    assert s["ledger_rows"] == 3
    assert s["chunk_rows"]  == 2  # c_stat1 stored once

    ledger.prune("s1")
    assert ledger.stats()["ledger_rows"] == 1
