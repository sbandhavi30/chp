"""Tests for WriteLimiter token-bucket rate limiting."""
from __future__ import annotations
import threading
import time
import pytest
from chp.ledger.lancedb_ledger import WriteLimiter, CHPLedger
from chp.schema.rationale_envelope import RationaleEnvelope


# ── helpers ───────────────────────────────────────────────────────────────────

def _env(cid: str) -> RationaleEnvelope:
    return RationaleEnvelope(
        chunk_id=cid, content=f"content {cid}",
        source_agent="agent-a", source_turn=0,
        hop_sequence=["agent-a", "agent-b"], selected_because=["test"],
        score=0.5, must_carry=False, token_cost=10, ledger_id=None,
    )


# ── constructor validation ────────────────────────────────────────────────────

def test_invalid_rate_raises():
    with pytest.raises(ValueError, match="rate"):
        WriteLimiter(rate=0)

def test_invalid_burst_raises():
    with pytest.raises(ValueError, match="burst"):
        WriteLimiter(rate=10, burst=0)


# ── token bucket mechanics ────────────────────────────────────────────────────

def test_burst_allows_up_to_burst_writes():
    limiter = WriteLimiter(rate=10, burst=5)
    # First 5 writes succeed (full bucket)
    for _ in range(5):
        limiter.check("agent-x")


def test_burst_exceeded_raises():
    limiter = WriteLimiter(rate=10, burst=5)
    for _ in range(5):
        limiter.check("agent-x")
    with pytest.raises(RuntimeError, match="rate limit"):
        limiter.check("agent-x")


def test_tokens_refill_over_time():
    limiter = WriteLimiter(rate=100, burst=2)
    limiter.check("agent-x")
    limiter.check("agent-x")
    # Bucket empty — sleep to refill 1 token (1/100 = 10ms)
    time.sleep(0.02)
    limiter.check("agent-x")   # should succeed after refill


def test_different_agents_independent_buckets():
    limiter = WriteLimiter(rate=10, burst=2)
    limiter.check("agent-a")
    limiter.check("agent-a")
    # agent-a exhausted — agent-b still has full bucket
    limiter.check("agent-b")
    limiter.check("agent-b")
    with pytest.raises(RuntimeError):
        limiter.check("agent-a")  # agent-a still exhausted


def test_stats_returns_token_levels():
    limiter = WriteLimiter(rate=10, burst=10)
    limiter.check("agent-x")
    limiter.check("agent-x")
    stats = limiter.stats()
    assert "agent-x" in stats
    # Started at 10, used 2 → ~8 (minus tiny refill in between)
    assert 7.0 <= stats["agent-x"] <= 10.0


def test_stats_empty_before_first_write():
    limiter = WriteLimiter(rate=10, burst=10)
    assert limiter.stats() == {}


def test_thread_safe_burst_enforcement():
    """50 threads each fire 4 writes against burst=100 — none should error."""
    limiter = WriteLimiter(rate=1000, burst=100)
    errors = []

    def worker():
        try:
            for _ in range(4):
                limiter.check("shared-agent")
        except RuntimeError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=worker) for _ in range(25)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # 25 threads × 4 = 100 writes == burst — all succeed
    assert not errors


# ── integration: CHPLedger + write_limiter ────────────────────────────────────

def test_ledger_write_limiter_blocks_runaway_agent():
    # rate=0.001 writes/s → refills 1 token per 1000 seconds — negligible during test
    limiter = WriteLimiter(rate=0.001, burst=3)
    ledger = CHPLedger(write_limiter=limiter)

    # 3 writes succeed (burst=3)
    for i in range(3):
        ledger.write("sess-rl", i, "agent-a", "b", _env(f"rl-c{i}"))

    # 4th write must raise rate-limit error (not duplicate error)
    with pytest.raises(RuntimeError, match="rate limit"):
        ledger.write("sess-rl", 3, "agent-a", "b", _env("rl-c3"))


def test_ledger_no_limiter_by_default():
    """Default CHPLedger has no rate limiting — many writes succeed."""
    ledger = CHPLedger()
    for i in range(50):
        ledger.write("sess-nolimit", i, "agent-a", "b", _env(f"nl-c{i}"))
    assert ledger.stats()["ledger_rows"] == 50


def test_ledger_per_agent_limiting():
    """Agent A hits limit; Agent B still writes freely."""
    limiter = WriteLimiter(rate=0.001, burst=2)
    ledger = CHPLedger(write_limiter=limiter)

    # agent-a exhausts burst
    ledger.write("sess-pa", 0, "agent-a", "b", _env("pa-c0"))
    ledger.write("sess-pa", 1, "agent-a", "b", _env("pa-c1"))
    with pytest.raises(RuntimeError, match="rate limit"):
        ledger.write("sess-pa", 2, "agent-a", "b", _env("pa-c2"))

    # agent-b unaffected (independent bucket)
    ledger.write("sess-pa", 0, "agent-b", "b", _env("pb-c0"))
    ledger.write("sess-pa", 1, "agent-b", "b", _env("pb-c1"))


def test_write_limiter_exported_from_package():
    import chp
    assert chp.WriteLimiter is WriteLimiter
