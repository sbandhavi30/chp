import pytest
from datetime import datetime, timezone
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.engine.embedder import StubEmbedder
from chp.engine.scorer import ScorerWeights, score_chunks, select_chunks


def _manifest(must_carry=None, domain_tags=None, token_budget=1000, on_missing="warn"):
    return ContextManifest(
        chp_version="0.1",
        agent_id="test-agent",
        task="test_task",
        requires=ContextRequirements(
            must_carry=must_carry or [],
            domain_tags=domain_tags or ["billing"],
            history_depth="full",
            recency_window=None,
            exclude=[],
        ),
        token_budget=token_budget,
        on_missing=on_missing,
    )


def _chunk(chunk_id, content, token_cost=50, source_turn=1):
    return AnnotatedChunk(
        chunk_id=chunk_id,
        content=content,
        token_cost=token_cost,
        source_agent="agent-a",
        source_turn=source_turn,
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )


def test_score_chunks_returns_sorted_descending():
    chunks = [_chunk("c1", "billing info"), _chunk("c2", "weather data")]
    manifest = _manifest(domain_tags=["billing"])
    scored = score_chunks(chunks, manifest, StubEmbedder())
    assert len(scored) == 2
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)


def test_select_chunks_respects_token_budget():
    chunks = [_chunk(f"c{i}", f"chunk {i}", token_cost=300) for i in range(5)]
    manifest = _manifest(token_budget=700)
    selected = select_chunks(chunks, manifest, StubEmbedder())
    assert sum(c.token_cost for c in selected) <= 700


def test_select_chunks_always_includes_must_carry():
    chunks = [
        _chunk("c1", "user_id is 42", token_cost=10),
        _chunk("c2", "irrelevant data", token_cost=10),
        _chunk("c3", "order_id is 99", token_cost=10),
    ]
    manifest = _manifest(must_carry=["user_id", "order_id"], token_budget=100)
    selected = select_chunks(chunks, manifest, StubEmbedder())
    ids = [c.chunk_id for c in selected]
    assert "c1" in ids
    assert "c3" in ids


def test_select_chunks_excludes_tagged_chunks():
    chunks = [
        _chunk("c1", "PII_raw: email@example.com", token_cost=10),
        _chunk("c2", "billing amount: $50", token_cost=10),
    ]
    manifest = ContextManifest(
        chp_version="0.1", agent_id="a", task="t",
        requires=ContextRequirements(
            must_carry=[], domain_tags=["billing"],
            history_depth="full", recency_window=None,
            exclude=["PII_raw"],
        ),
        token_budget=1000, on_missing="warn",
    )
    selected = select_chunks(chunks, manifest, StubEmbedder())
    assert "c1" not in [c.chunk_id for c in selected]


def test_select_chunks_fail_hard_on_missing_must_carry():
    chunks = [_chunk("c1", "irrelevant", token_cost=10)]
    manifest = _manifest(must_carry=["user_id"], on_missing="fail_hard")
    with pytest.raises(ValueError, match="must_carry"):
        select_chunks(chunks, manifest, StubEmbedder())


# ── accept_upstream_output ────────────────────────────────────────────────────

def _output_chunk(chunk_id, content, source_agent="agent-a", token_cost=50):
    return AnnotatedChunk(
        chunk_id=chunk_id, content=content, token_cost=token_cost,
        source_agent=source_agent, source_turn=1,
        is_agent_output=True,
    )


def _manifest_with_upstream(accept, token_budget=1000, exclude=None):
    return ContextManifest(
        chp_version="0.1", agent_id="agent-b", task="t",
        requires=ContextRequirements(
            must_carry=[], domain_tags=["billing"],
            history_depth="full", exclude=exclude or [],
            accept_upstream_output=accept,
        ),
        token_budget=token_budget, on_missing="warn",
    )


def test_upstream_output_excluded_when_flag_false():
    """Default False: agent-b ignores agent-a's output."""
    chunks = [_output_chunk("out-1", "routing decision: priority=HIGH")]
    manifest = _manifest_with_upstream(accept=False)
    selected = select_chunks(chunks, manifest, StubEmbedder())
    assert "out-1" not in [c.chunk_id for c in selected]


def test_upstream_output_included_when_flag_true():
    """True: agent-b accepts output from any upstream agent."""
    chunks = [_output_chunk("out-1", "routing decision: priority=HIGH")]
    manifest = _manifest_with_upstream(accept=True)
    selected = select_chunks(chunks, manifest, StubEmbedder())
    assert "out-1" in [c.chunk_id for c in selected]


def test_upstream_output_included_when_agent_listed():
    """List: agent-b accepts only from agent-a, not agent-c."""
    chunks = [
        _output_chunk("out-a", "agent-a result", source_agent="agent-a"),
        _output_chunk("out-c", "agent-c result", source_agent="agent-c"),
    ]
    manifest = _manifest_with_upstream(accept=["agent-a"])
    selected = select_chunks(chunks, manifest, StubEmbedder())
    ids = [c.chunk_id for c in selected]
    assert "out-a" in ids
    assert "out-c" not in ids


def test_upstream_output_blocked_by_exclude():
    """accept=True but exclude fires — PII in A's output never reaches B."""
    chunks = [_output_chunk("out-pii", "PII_raw: ssn=123-45-6789", source_agent="agent-a")]
    manifest = _manifest_with_upstream(accept=True, exclude=["PII_raw"])
    selected = select_chunks(chunks, manifest, StubEmbedder())
    assert "out-pii" not in [c.chunk_id for c in selected]


def test_upstream_output_capped_at_20_percent_budget():
    """Agent output capped at 20% of token_budget — A can't flood B."""
    # budget=1000, cap=200. Each output chunk = 150 tokens → only 1 fits.
    chunks = [
        _output_chunk(f"out-{i}", f"agent output {i}", token_cost=150)
        for i in range(5)
    ]
    manifest = _manifest_with_upstream(accept=True, token_budget=1000)
    selected = select_chunks(chunks, manifest, StubEmbedder())
    output_selected = [c for c in selected if c.is_agent_output]
    assert sum(c.token_cost for c in output_selected) <= 200


def test_upstream_output_does_not_eat_must_carry_budget():
    """Upstream output budget is separate — must_carry still delivered."""
    pool_chunk = AnnotatedChunk(
        chunk_id="must-c", content="order_id: ORD-42",
        token_cost=100, source_agent="router", source_turn=1,
    )
    output = _output_chunk("out-1", "routing decision", token_cost=150)
    manifest = ContextManifest(
        chp_version="0.1", agent_id="agent-b", task="t",
        requires=ContextRequirements(
            must_carry=["order_id"], domain_tags=["billing"],
            history_depth="full", exclude=[],
            accept_upstream_output=True,
        ),
        token_budget=500, on_missing="warn",
    )
    selected = select_chunks([pool_chunk, output], manifest, StubEmbedder())
    ids = [c.chunk_id for c in selected]
    assert "must-c" in ids


def test_normal_chunks_not_treated_as_agent_output():
    """Chunks with is_agent_output=False go through normal scoring regardless of flag."""
    chunk = _chunk("c1", "billing info", token_cost=50)
    # accept=False should not block normal pool chunks
    manifest = _manifest_with_upstream(accept=False, token_budget=200)
    selected = select_chunks([chunk], manifest, StubEmbedder())
    assert "c1" in [c.chunk_id for c in selected]


# ── ledger_fallback ───────────────────────────────────────────────────────────

from chp.ledger.lancedb_ledger import CHPLedger
from chp.schema.rationale_envelope import RationaleEnvelope


def _ledger_manifest(must_carry, exclude=None, token_budget=1000):
    return ContextManifest(
        chp_version="0.1", agent_id="fraud-agent", task="assess_fraud",
        requires=ContextRequirements(
            must_carry=must_carry,
            domain_tags=["fraud"],
            history_depth="decisions_only",
            exclude=exclude or [],
        ),
        token_budget=token_budget,
        on_missing="ledger_fallback",
    )


def _write_envelope(ledger, session_id, chunk_id, content, source_agent="billing-agent"):
    env = RationaleEnvelope(
        chunk_id=chunk_id, content=content,
        source_agent=source_agent, source_turn=1,
        hop_sequence=[source_agent, "fraud-agent"],
        selected_because=["domain_match"],
        score=0.9, must_carry=False, token_cost=80,
    )
    ledger.write(session_id, 0, source_agent, "fraud-agent", env)


def test_ledger_fallback_recovers_missing_must_carry():
    """Billing wrote output in hop 0; fraud skipped billing but needs billing_decision."""
    ledger = CHPLedger()
    session = "sess-fallback-001"
    _write_envelope(ledger, session, "billing-out", "billing_decision: approve refund $299")

    manifest = _ledger_manifest(must_carry=["billing_decision"])
    # Pool is empty — billing never passed to fraud directly
    selected = select_chunks([], manifest, StubEmbedder(), ledger=ledger, session_id=session)
    ids = [c.chunk_id for c in selected]
    assert "billing-out" in ids, "ledger_fallback must inject billing output"


def test_ledger_fallback_exclude_fires_on_recovered_chunk():
    """Even recovered chunks go through exclude — PII in billing output never reaches fraud."""
    ledger = CHPLedger()
    session = "sess-fallback-pii"
    _write_envelope(ledger, session, "billing-pii", "billing_decision: PII_raw ssn=123 approve")

    manifest = _ledger_manifest(must_carry=["billing_decision"], exclude=["PII_raw"])
    selected = select_chunks([], manifest, StubEmbedder(), ledger=ledger, session_id=session)
    assert "billing-pii" not in [c.chunk_id for c in selected]


def test_ledger_fallback_warns_when_ledger_has_nothing(caplog):
    """Ledger is empty — fallback finds nothing, logs warning, proceeds without."""
    import logging
    ledger = CHPLedger()
    session = "sess-fallback-empty"

    manifest = _ledger_manifest(must_carry=["billing_decision"])
    with caplog.at_level(logging.WARNING, logger="chp.engine.scorer"):
        selected = select_chunks([], manifest, StubEmbedder(), ledger=ledger, session_id=session)
    assert selected == []
    assert "billing_decision" in caplog.text


def test_ledger_fallback_degrades_to_warn_when_no_ledger_provided(caplog):
    """on_missing=ledger_fallback but ledger=None → warns, doesn't crash."""
    import logging
    manifest = _ledger_manifest(must_carry=["billing_decision"])
    with caplog.at_level(logging.WARNING, logger="chp.engine.scorer"):
        selected = select_chunks([], manifest, StubEmbedder())  # no ledger arg
    assert selected == []
    assert "billing_decision" in caplog.text


def test_ledger_fallback_pool_chunk_takes_priority():
    """If pool already has must_carry, fallback never fires."""
    ledger = CHPLedger()
    session = "sess-fallback-pool"
    # Write a DIFFERENT content to ledger — pool chunk should win
    _write_envelope(ledger, session, "ledger-out", "billing_decision: from ledger")

    pool_chunk = _chunk("pool-billing", "billing_decision: from pool approve $50", token_cost=40)
    manifest = _ledger_manifest(must_carry=["billing_decision"])
    selected = select_chunks([pool_chunk], manifest, StubEmbedder(), ledger=ledger, session_id=session)
    ids = [c.chunk_id for c in selected]
    assert "pool-billing" in ids
    assert "ledger-out" not in ids  # pool took priority, fallback didn't fire


def test_ledger_fallback_end_to_end_skipped_agent():
    """
    Full pipeline: billing runs hop 0, fraud skips billing (jumps directly),
    fraud's manifest uses ledger_fallback → recovers billing output automatically.
    """
    ledger = CHPLedger()
    session = "sess-e2e-skip"

    # Billing agent ran at hop 0, wrote output to ledger
    billing_env = RationaleEnvelope(
        chunk_id="billing-decision-001",
        content="billing_decision: approve refund $299 tier=premium",
        source_agent="billing-agent", source_turn=1,
        hop_sequence=["router", "billing-agent"],
        selected_because=["must_carry:order_id"],
        score=0.95, must_carry=True, token_cost=60,
    )
    ledger.write(session, 0, "router", "billing-agent", billing_env)

    # Orchestrator decided to skip billing→fraud chain, jump straight to fraud
    # Fraud's pool has NO billing output
    fraud_pool = [
        _chunk("fraud-signals", "fraud_score: 0.12 device: mobile velocity: low", token_cost=80),
    ]
    fraud_manifest = ContextManifest(
        chp_version="0.1", agent_id="fraud-agent", task="assess_fraud",
        requires=ContextRequirements(
            must_carry=["billing_decision", "fraud_score"],
            domain_tags=["fraud", "billing"],
            history_depth="decisions_only",
            exclude=["PII_raw", "ssn"],
        ),
        token_budget=800,
        on_missing="ledger_fallback",
    )

    selected = select_chunks(
        fraud_pool, fraud_manifest, StubEmbedder(),
        ledger=ledger, session_id=session,
    )
    ids = [c.chunk_id for c in selected]

    # fraud_score came from pool, billing_decision recovered from ledger
    assert "fraud-signals" in ids,        "fraud_score must come from pool"
    assert "billing-decision-001" in ids, "billing_decision must be recovered from ledger"


# ── Case B: parallel race — retry window ─────────────────────────────────────

def test_ledger_fallback_retry_succeeds_when_writer_arrives_late():
    """
    Billing and fraud dispatched in parallel. Fraud starts before billing writes.
    With retry_attempts=3, fraud polls ledger and recovers billing output once it lands.
    """
    import threading, time
    ledger = CHPLedger()
    session = "sess-race-001"

    def write_after_delay():
        time.sleep(0.15)   # 150ms — billing "arrives late"
        _write_envelope(ledger, session, "billing-race", "billing_decision: approve $299")

    writer = threading.Thread(target=write_after_delay)
    writer.start()

    manifest = ContextManifest(
        chp_version="0.1", agent_id="fraud-agent", task="assess_fraud",
        requires=ContextRequirements(
            must_carry=["billing_decision"],
            domain_tags=["fraud"],
            history_depth="decisions_only",
            exclude=[],
            fallback_retry_attempts=3,
            fallback_retry_delay_ms=100,   # poll every 100ms, billing arrives at 150ms
        ),
        token_budget=800,
        on_missing="ledger_fallback",
    )
    selected = select_chunks([], manifest, StubEmbedder(), ledger=ledger, session_id=session)
    writer.join()

    assert "billing-race" in [c.chunk_id for c in selected], \
        "retry must recover billing output that arrived after fraud started"


def test_ledger_fallback_retry_gives_up_after_max_attempts(caplog):
    """If writer never arrives, fallback logs warning after exhausting retries."""
    import logging
    ledger = CHPLedger()
    session = "sess-race-timeout"

    manifest = ContextManifest(
        chp_version="0.1", agent_id="fraud-agent", task="t",
        requires=ContextRequirements(
            must_carry=["billing_decision"],
            domain_tags=["fraud"],
            history_depth="decisions_only",
            exclude=[],
            fallback_retry_attempts=2,
            fallback_retry_delay_ms=10,   # short for test speed
        ),
        token_budget=800,
        on_missing="ledger_fallback",
    )
    with caplog.at_level(logging.WARNING, logger="chp.engine.scorer"):
        selected = select_chunks([], manifest, StubEmbedder(), ledger=ledger, session_id=session)

    assert selected == []
    assert "billing_decision" in caplog.text


def test_ledger_fallback_zero_retries_does_not_sleep():
    """retry_attempts=0 (default) must not introduce any sleep delay."""
    import time
    ledger = CHPLedger()
    session = "sess-no-retry"

    manifest = ContextManifest(
        chp_version="0.1", agent_id="fraud-agent", task="t",
        requires=ContextRequirements(
            must_carry=["billing_decision"],
            domain_tags=["fraud"],
            history_depth="decisions_only",
            exclude=[],
            fallback_retry_attempts=0,
            fallback_retry_delay_ms=5000,   # large — must never be hit
        ),
        token_budget=800,
        on_missing="ledger_fallback",
    )
    t0 = time.monotonic()
    select_chunks([], manifest, StubEmbedder(), ledger=ledger, session_id=session)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, f"zero retries must not sleep; took {elapsed:.2f}s"


# ── Case C: cross-session — parent_session_id ─────────────────────────────────

def test_ledger_fallback_cross_session_recovers_from_parent():
    """
    Billing ran in session-001 (completed). session-002 fraud needs billing output.
    parent_session_id links them — fraud recovers from parent session.
    """
    ledger = CHPLedger()
    parent_session = "sess-parent-001"
    child_session  = "sess-child-002"

    # Billing ran and wrote in the PARENT session
    _write_envelope(ledger, parent_session, "billing-parent", "billing_decision: approve $299")

    # Child session has nothing from billing
    manifest = ContextManifest(
        chp_version="0.1", agent_id="fraud-agent", task="assess_fraud",
        requires=ContextRequirements(
            must_carry=["billing_decision"],
            domain_tags=["fraud"],
            history_depth="decisions_only",
            exclude=[],
        ),
        token_budget=800,
        on_missing="ledger_fallback",
        parent_session_id=parent_session,
    )
    selected = select_chunks([], manifest, StubEmbedder(), ledger=ledger, session_id=child_session)
    assert "billing-parent" in [c.chunk_id for c in selected], \
        "cross-session fallback must recover billing output from parent session"


def test_ledger_fallback_no_cross_session_bleed_without_parent_session_id(caplog):
    """Without parent_session_id, other sessions' data never bleeds in."""
    import logging
    ledger = CHPLedger()
    unrelated_session = "sess-unrelated-999"
    child_session     = "sess-child-isolated"

    # Some other session has billing data — must NOT reach child
    _write_envelope(ledger, unrelated_session, "billing-unrelated", "billing_decision: other customer")

    manifest = ContextManifest(
        chp_version="0.1", agent_id="fraud-agent", task="t",
        requires=ContextRequirements(
            must_carry=["billing_decision"],
            domain_tags=["fraud"],
            history_depth="decisions_only",
            exclude=[],
        ),
        token_budget=800,
        on_missing="ledger_fallback",
        # parent_session_id intentionally absent
    )
    with caplog.at_level(logging.WARNING, logger="chp.engine.scorer"):
        selected = select_chunks([], manifest, StubEmbedder(), ledger=ledger, session_id=child_session)

    assert "billing-unrelated" not in [c.chunk_id for c in selected], \
        "unrelated session data must never bleed into child session"


def test_ledger_fallback_parent_session_exclude_still_fires():
    """PII in parent session data is still blocked by exclude — even in cross-session recovery."""
    ledger = CHPLedger()
    parent_session = "sess-parent-pii"
    child_session  = "sess-child-pii"

    _write_envelope(ledger, parent_session, "billing-pii-parent",
                    "billing_decision: PII_raw ssn=123 approve")

    manifest = ContextManifest(
        chp_version="0.1", agent_id="fraud-agent", task="t",
        requires=ContextRequirements(
            must_carry=["billing_decision"],
            domain_tags=["fraud"],
            history_depth="decisions_only",
            exclude=["PII_raw"],
        ),
        token_budget=800,
        on_missing="ledger_fallback",
        parent_session_id=parent_session,
    )
    selected = select_chunks([], manifest, StubEmbedder(), ledger=ledger, session_id=child_session)
    assert "billing-pii-parent" not in [c.chunk_id for c in selected], \
        "exclude must fire even on cross-session recovered chunks"


def test_ledger_fallback_current_session_wins_over_parent():
    """Current session has billing output AND parent does too — current wins."""
    ledger = CHPLedger()
    parent_session = "sess-parent-both"
    child_session  = "sess-child-both"

    _write_envelope(ledger, parent_session, "billing-from-parent", "billing_decision: from parent")
    _write_envelope(ledger, child_session,  "billing-from-child",  "billing_decision: from child")

    manifest = ContextManifest(
        chp_version="0.1", agent_id="fraud-agent", task="t",
        requires=ContextRequirements(
            must_carry=["billing_decision"],
            domain_tags=["fraud"],
            history_depth="decisions_only",
            exclude=[],
        ),
        token_budget=800,
        on_missing="ledger_fallback",
        parent_session_id=parent_session,
    )
    selected = select_chunks([], manifest, StubEmbedder(), ledger=ledger, session_id=child_session)
    ids = [c.chunk_id for c in selected]
    assert "billing-from-child" in ids,  "current session must take priority"
    assert "billing-from-parent" not in ids, "parent not queried when current session satisfied"
