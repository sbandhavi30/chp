"""
Backend-agnostic contract tests — same suite runs against all LedgerBackend impls.

Parametrize over SQLiteLedger and InMemoryLedger (no LanceDB dep needed).
LanceDBLedger already has its own dedicated test files.
"""
from __future__ import annotations
import threading
import pytest
from chp.ledger.sqlite_ledger import SQLiteLedger
from chp.ledger.memory_ledger import InMemoryLedger
from chp.ledger.base import LedgerBackend
from chp.schema.rationale_envelope import RationaleEnvelope
import chp


# ── fixture ───────────────────────────────────────────────────────────────────

def _env(cid: str, content: str = "", agent: str = "agent-a", cost: int = 10) -> RationaleEnvelope:
    return RationaleEnvelope(
        chunk_id=cid,
        content=content or f"content for {cid}",
        source_agent=agent,
        source_turn=1,
        hop_sequence=[agent, "agent-b"],
        selected_because=["test"],
        score=0.5,
        must_carry=False,
        token_cost=cost,
        ledger_id=None,
    )


@pytest.fixture(params=["sqlite", "memory"])
def ledger(request, tmp_path) -> LedgerBackend:
    if request.param == "sqlite":
        return SQLiteLedger(str(tmp_path / "test.db"))
    return InMemoryLedger()


# ── contract: write + query ───────────────────────────────────────────────────

def test_write_returns_ledger_id(ledger):
    lid = ledger.write("sess-1", 0, "agent-a", "agent-b", _env("c1"))
    assert isinstance(lid, str) and lid


def test_query_returns_written_envelope(ledger):
    ledger.write("sess-1", 0, "agent-a", "agent-b", _env("c1", "hello world"))
    rows = ledger.query("sess-1")
    assert len(rows) == 1
    assert rows[0].chunk_id == "c1"
    assert rows[0].content == "hello world"
    assert rows[0].source_agent == "agent-a"
    assert rows[0].token_cost == 10


def test_query_filters_by_agent(ledger):
    ledger.write("sess-1", 0, "agent-a", "agent-b", _env("c1"))
    ledger.write("sess-1", 1, "agent-a", "agent-c", _env("c2"))
    rows = ledger.query("sess-1", agent_id="agent-b")
    assert len(rows) == 1
    assert rows[0].chunk_id == "c1"


def test_query_different_sessions_isolated(ledger):
    ledger.write("sess-A", 0, "agent-a", "agent-b", _env("c1"))
    ledger.write("sess-B", 0, "agent-a", "agent-b", _env("c2"))
    assert len(ledger.query("sess-A")) == 1
    assert len(ledger.query("sess-B")) == 1
    assert ledger.query("sess-A")[0].chunk_id == "c1"


def test_query_empty_session(ledger):
    assert ledger.query("nonexistent-session") == []


def test_duplicate_raises_runtime_error(ledger):
    ledger.write("sess-1", 0, "agent-a", "agent-b", _env("c1"))
    with pytest.raises(RuntimeError, match="duplicate"):
        ledger.write("sess-1", 0, "agent-a", "agent-b", _env("c1"))


def test_same_chunk_different_hop_allowed(ledger):
    ledger.write("sess-1", 0, "agent-a", "agent-b", _env("c1"))
    ledger.write("sess-1", 1, "agent-a", "agent-b", _env("c1"))  # hop=1, not duplicate
    assert len(ledger.query("sess-1")) == 2


def test_chunk_content_deduplicated(ledger):
    ledger.write("sess-1", 0, "agent-a", "agent-b", _env("c1", "shared content"))
    ledger.write("sess-2", 0, "agent-a", "agent-b", _env("c1", "shared content"))
    stats = ledger.stats()
    # c1 stored once even across sessions
    assert stats["chunk_rows"] == 1
    assert stats["ledger_rows"] == 2


# ── contract: query_hop ───────────────────────────────────────────────────────

def test_query_hop_returns_correct_hop(ledger):
    ledger.write("sess-1", 0, "agent-a", "agent-b", _env("c1"))
    ledger.write("sess-1", 1, "agent-a", "agent-b", _env("c2"))
    ledger.write("sess-1", 2, "agent-a", "agent-b", _env("c3"))
    hop1 = ledger.query_hop("sess-1", 1)
    assert len(hop1) == 1
    assert hop1[0].chunk_id == "c2"


# ── contract: prune ───────────────────────────────────────────────────────────

def test_prune_removes_session_rows(ledger):
    ledger.write("sess-1", 0, "agent-a", "agent-b", _env("c1"))
    ledger.write("sess-2", 0, "agent-a", "agent-b", _env("c2"))
    deleted = ledger.prune("sess-1")
    assert deleted == 1
    assert ledger.query("sess-1") == []
    assert len(ledger.query("sess-2")) == 1


def test_prune_orphan_chunks(ledger):
    ledger.write("sess-1", 0, "agent-a", "agent-b", _env("c1"))
    ledger.prune("sess-1")
    orphans = ledger.prune_orphan_chunks()
    assert orphans == 1
    assert ledger.stats()["chunk_rows"] == 0


def test_prune_older_than(ledger):
    ledger.write("sess-1", 0, "agent-a", "agent-b", _env("c1"))
    ledger.write("sess-2", 0, "agent-a", "agent-b", _env("c2"))
    # Use far-future cutoff — should delete everything
    deleted = ledger.prune_older_than("2099-01-01T00:00:00+00:00")
    assert deleted == 2
    assert ledger.stats()["ledger_rows"] == 0


# ── contract: stats ───────────────────────────────────────────────────────────

def test_stats_reflects_writes(ledger):
    assert ledger.stats()["ledger_rows"] == 0
    assert ledger.stats()["chunk_rows"] == 0
    ledger.write("sess-1", 0, "agent-a", "agent-b", _env("c1"))
    ledger.write("sess-1", 1, "agent-a", "agent-b", _env("c2"))
    assert ledger.stats()["ledger_rows"] == 2
    assert ledger.stats()["chunk_rows"] == 2


# ── contract: security — safe_id injection guard ─────────────────────────────

@pytest.mark.parametrize("bad_id", ["'; DROP TABLE chp_ledger; --", "a b", "", "a/b"])
def test_invalid_session_id_rejected(ledger, bad_id):
    with pytest.raises((ValueError, RuntimeError)):
        ledger.write(bad_id, 0, "agent-a", "agent-b", _env("c1"))


# ── contract: async API ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_awrite_aquery(ledger):
    await ledger.awrite("sess-async", 0, "agent-a", "agent-b", _env("ac1", "async content"))
    rows = await ledger.aquery("sess-async")
    assert len(rows) == 1
    assert rows[0].content == "async content"


@pytest.mark.asyncio
async def test_aquery_hop(ledger):
    await ledger.awrite("sess-ah", 0, "agent-a", "agent-b", _env("ah1"))
    await ledger.awrite("sess-ah", 1, "agent-a", "agent-b", _env("ah2"))
    rows = await ledger.aquery_hop("sess-ah", 0)
    assert len(rows) == 1
    assert rows[0].chunk_id == "ah1"


@pytest.mark.asyncio
async def test_aprune(ledger):
    await ledger.awrite("sess-ap", 0, "agent-a", "agent-b", _env("ap1"))
    deleted = await ledger.aprune("sess-ap")
    assert deleted == 1


# ── contract: query_by_meaning returns [] for non-vector backends ─────────────

def test_query_by_meaning_returns_empty_list(ledger):
    from chp.engine.embedder import StubEmbedder
    ledger.write("sess-1", 0, "agent-a", "agent-b", _env("c1", "some content"))
    result = ledger.query_by_meaning("some content", StubEmbedder())
    assert result == []   # SQLite and InMemory don't support ANN — graceful []


# ── InMemoryLedger-specific ────────────────────────────────────────────────────

def test_inmemory_clear():
    mem = InMemoryLedger()
    mem.write("sess-1", 0, "a", "b", _env("c1"))
    mem.clear()
    assert mem.stats()["ledger_rows"] == 0
    assert mem.stats()["chunk_rows"] == 0


def test_inmemory_thread_safe():
    mem = InMemoryLedger()
    errors = []

    def worker(idx):
        try:
            mem.write(f"sess-{idx}", 0, "a", "b", _env(f"c-{idx}"))
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert mem.stats()["ledger_rows"] == 50


# ── SQLiteLedger-specific ──────────────────────────────────────────────────────

def test_sqlite_persistent(tmp_path):
    path = str(tmp_path / "persist.db")
    ledger1 = SQLiteLedger(path)
    ledger1.write("sess-1", 0, "a", "b", _env("c1", "persisted content"))
    ledger1.close()

    ledger2 = SQLiteLedger(path)
    rows = ledger2.query("sess-1")
    assert len(rows) == 1
    assert rows[0].content == "persisted content"
    ledger2.close()


def test_sqlite_compact_does_not_raise(tmp_path):
    ledger = SQLiteLedger(str(tmp_path / "compact.db"))
    for i in range(5):
        ledger.write("sess-1", i, "a", "b", _env(f"c{i}"))
    ledger.prune("sess-1")
    ledger.compact()   # VACUUM — must not raise


# ── Package-level exports ─────────────────────────────────────────────────────

def test_backends_exported_from_package():
    assert chp.LedgerBackend is LedgerBackend
    assert chp.SQLiteLedger is SQLiteLedger
    assert chp.InMemoryLedger is InMemoryLedger
    assert chp.LanceDBLedger is not None
    assert chp.CHPLedger is chp.LanceDBLedger   # backward-compat alias
