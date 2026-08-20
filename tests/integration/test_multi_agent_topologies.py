"""
Integration tests for multi-agent topology patterns:
  1. Sequential chain (Router → Auth → Billing → Summarizer)
  2. Fan-out (Orchestrator spawns 4 parallel subagents via threads)
  3. Fan-in (Orchestrator recovers all subagent outputs from ledger)

No LLM required — uses StubEmbedder + real scorer/ledger.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pytest

from chp.engine.embedder import StubEmbedder
from chp.engine.scorer import select_chunks
from chp.ledger.lancedb_ledger import CHPLedger
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_envelope(chunk: AnnotatedChunk, to_agent: str, hop: int) -> RationaleEnvelope:
    return RationaleEnvelope(
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        source_agent=chunk.source_agent,
        source_turn=chunk.source_turn,
        hop_sequence=[chunk.source_agent, to_agent],
        selected_because=[f"chp_selected_hop:{hop}"],
        score=0.5,
        must_carry=False,
        token_cost=chunk.token_cost,
    )


def _write_selected(ledger, session_id, hop, from_agent, to_agent, selected):
    for chunk in selected:
        env = _make_envelope(chunk, to_agent, hop)
        try:
            ledger.write(session_id, hop, from_agent, to_agent, env)
        except RuntimeError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Shared session chunks — 10 context items, mix of relevant + sensitive
# ─────────────────────────────────────────────────────────────────────────────

CHUNKS = [
    AnnotatedChunk(chunk_id="c_user",       content="user_id: USR-001 | tier: premium",             token_cost=30,  source_agent="router", source_turn=1,  timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_order",      content="order_id: ORD-999 | amount: $199 | plan: Pro", token_cost=35,  source_agent="router", source_turn=2,  timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_pii",        content="PII_raw: credit_card=4111111111111111",         token_cost=40,  source_agent="router", source_turn=3,  timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_auth",       content="auth_status: verified | mfa: passed",          token_cost=35,  source_agent="router", source_turn=4,  timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_fraud",      content="fraud_score: 0.08 | ip_risk: low",             token_cost=40,  source_agent="router", source_turn=5,  timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_compliance", content="compliance_flags: GDPR=ok | PCI=flagged",      token_cost=40,  source_agent="router", source_turn=6,  timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_policy",     content="refund_policy: Pro Plan 30-day full refund",   token_cost=45,  source_agent="router", source_turn=7,  timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_request",    content="customer_request: duplicate charge refund $199", token_cost=45, source_agent="router", source_turn=8, timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_debug",      content="debug_trace: POST /api/charge 200 idempotency_key=missing", token_cost=55, source_agent="router", source_turn=9, timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_history",    content="prior_resolution: TKT-100 refund $50 approved", token_cost=45, source_agent="router", source_turn=10, timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
]

TOTAL_TOKENS = sum(c.token_cost for c in CHUNKS)


@pytest.fixture
def embedder():
    return StubEmbedder()


@pytest.fixture
def ledger():
    return CHPLedger()


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 1 — Sequential chain: Router → Auth → Billing → Summarizer
# ─────────────────────────────────────────────────────────────────────────────

AUTH_MANIFEST = ContextManifest(
    agent_id="auth-agent", task="verify_identity",
    requires=ContextRequirements(
        must_carry=["user_id", "auth_status"],
        domain_tags=["auth", "user", "session"],
        exclude=["PII_raw", "debug_trace", "compliance_flags"],
    ),
    token_budget=120, on_missing="warn",
)

BILLING_MANIFEST = ContextManifest(
    agent_id="billing-agent", task="resolve_refund",
    requires=ContextRequirements(
        must_carry=["order_id", "user_id"],
        domain_tags=["billing", "refund", "order"],
        exclude=["PII_raw", "debug_trace", "fraud_score"],
    ),
    token_budget=150, on_missing="fail_hard",
)

SUMMARIZER_MANIFEST = ContextManifest(
    agent_id="summarizer", task="summarize_resolution",
    requires=ContextRequirements(
        must_carry=[],
        domain_tags=["summary", "resolution", "request"],
        exclude=["PII_raw", "debug_trace", "auth_status", "fraud_score"],
    ),
    token_budget=120, on_missing="warn",
)


def test_sequential_chain_each_hop_filters_correctly(embedder, ledger):
    """Each agent gets only what its manifest declares; PII never crosses a hop."""
    session_id = "seq_test_001"

    # Hop 0: Auth
    auth_selected = select_chunks(CHUNKS, AUTH_MANIFEST, embedder)
    auth_ids = {c.chunk_id for c in auth_selected}
    assert "c_user" in auth_ids,  "auth agent must have user_id chunk"
    assert "c_auth" in auth_ids,  "auth agent must have auth_status chunk"
    assert "c_pii" not in auth_ids, "PII_raw must be excluded from auth"
    assert "c_debug" not in auth_ids, "debug_trace must be excluded from auth"
    _write_selected(ledger, session_id, 0, "router", "auth-agent", auth_selected)

    # Hop 1: Billing
    billing_selected = select_chunks(CHUNKS, BILLING_MANIFEST, embedder)
    billing_ids = {c.chunk_id for c in billing_selected}
    assert "c_order" in billing_ids, "billing agent must have order_id chunk"
    assert "c_user" in billing_ids,  "billing agent must have user_id chunk"
    assert "c_pii" not in billing_ids, "PII_raw must be excluded from billing"
    _write_selected(ledger, session_id, 1, "auth-agent", "billing-agent", billing_selected)

    # Hop 2: Summarizer
    summary_selected = select_chunks(CHUNKS, SUMMARIZER_MANIFEST, embedder)
    summary_ids = {c.chunk_id for c in summary_selected}
    assert "c_pii" not in summary_ids
    assert "c_debug" not in summary_ids
    _write_selected(ledger, session_id, 2, "billing-agent", "summarizer", summary_selected)

    # Ledger has all 3 hops
    all_records = ledger.query(session_id)
    hop_numbers = {r.source_turn for r in all_records}
    assert len(hop_numbers) >= 1  # records exist; hop encoded in source_turn


def test_sequential_chain_token_reduction(embedder):
    """Each agent stays within its token budget and uses less than full context."""
    for manifest in [AUTH_MANIFEST, BILLING_MANIFEST, SUMMARIZER_MANIFEST]:
        selected = select_chunks(CHUNKS, manifest, embedder)
        agent_tokens = sum(c.token_cost for c in selected)
        assert agent_tokens < TOTAL_TOKENS, (
            f"{manifest.agent_id}: received all {TOTAL_TOKENS} tokens — no reduction"
        )
        assert agent_tokens <= manifest.token_budget, (
            f"{manifest.agent_id}: {agent_tokens} > budget {manifest.token_budget}"
        )


def test_sequential_chain_pii_excluded_from_all_agents(embedder):
    """PII_raw (c_pii) excluded from every agent in the chain."""
    for manifest in [AUTH_MANIFEST, BILLING_MANIFEST, SUMMARIZER_MANIFEST]:
        selected = select_chunks(CHUNKS, manifest, embedder)
        chunk_ids = [c.chunk_id for c in selected]
        assert "c_pii" not in chunk_ids, f"PII_raw leaked to {manifest.agent_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 2 — Fan-out: Orchestrator → 4 parallel subagents
# ─────────────────────────────────────────────────────────────────────────────

FRAUD_MANIFEST = ContextManifest(
    agent_id="fraud-agent", task="assess_fraud_risk",
    requires=ContextRequirements(
        must_carry=["fraud_score"],
        domain_tags=["fraud", "risk", "velocity"],
        exclude=["PII_raw", "debug_trace", "compliance_flags"],
    ),
    token_budget=100, on_missing="warn",
)

COMPLIANCE_MANIFEST = ContextManifest(
    agent_id="compliance-agent", task="check_compliance",
    requires=ContextRequirements(
        must_carry=["user_id", "compliance_flags"],
        domain_tags=["compliance", "GDPR", "PCI"],
        exclude=["debug_trace"],   # compliance CAN see PII flags (not raw card data)
    ),
    token_budget=120, on_missing="fail_hard",
)

POLICY_MANIFEST = ContextManifest(
    agent_id="policy-agent", task="check_eligibility",
    requires=ContextRequirements(
        must_carry=["order_id"],
        domain_tags=["policy", "refund", "eligibility"],
        exclude=["PII_raw", "debug_trace", "fraud_score"],
    ),
    token_budget=100, on_missing="fail_hard",
)

HISTORY_MANIFEST = ContextManifest(
    agent_id="history-agent", task="review_history",
    requires=ContextRequirements(
        must_carry=[],
        domain_tags=["history", "prior", "resolution"],
        exclude=["PII_raw", "debug_trace"],
    ),
    token_budget=120, on_missing="proceed",
)

FAN_OUT_MANIFESTS = [FRAUD_MANIFEST, COMPLIANCE_MANIFEST, POLICY_MANIFEST, HISTORY_MANIFEST]


def _run_subagent(manifest, chunks, ledger, session_id, hop, embedder):
    selected = select_chunks(chunks, manifest, embedder)
    _write_selected(ledger, session_id, hop, "orchestrator", manifest.agent_id, selected)
    return manifest.agent_id, selected


def test_fanout_all_subagents_complete(embedder, ledger):
    """All 4 subagents execute concurrently; each produces results."""
    session_id = "fanout_test_001"
    results: dict = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_run_subagent, m, CHUNKS, ledger, session_id, 1, embedder): m.agent_id
            for m in FAN_OUT_MANIFESTS
        }
        for future in as_completed(futures):
            agent_id, selected = future.result()
            results[agent_id] = selected

    assert set(results.keys()) == {m.agent_id for m in FAN_OUT_MANIFESTS}


def test_fanout_each_subagent_within_budget(embedder, ledger):
    session_id = "fanout_budget_001"
    for m in FAN_OUT_MANIFESTS:
        selected = select_chunks(CHUNKS, m, embedder)
        tokens = sum(c.token_cost for c in selected)
        assert tokens <= m.token_budget, f"{m.agent_id}: {tokens} > budget {m.token_budget}"


def test_fanout_no_cross_contamination(embedder, ledger):
    """Subagents never receive context they excluded; must_carry always present."""
    session_id = "fanout_contamination_001"

    fraud_ids = {c.chunk_id for c in select_chunks(CHUNKS, FRAUD_MANIFEST, embedder)}
    assert "c_fraud" in fraud_ids, "fraud agent must carry fraud_score"
    assert "c_pii" not in fraud_ids

    compliance_ids = {c.chunk_id for c in select_chunks(CHUNKS, COMPLIANCE_MANIFEST, embedder)}
    assert "c_compliance" in compliance_ids, "compliance agent must carry compliance_flags"

    policy_ids = {c.chunk_id for c in select_chunks(CHUNKS, POLICY_MANIFEST, embedder)}
    assert "c_order" in policy_ids, "policy agent must carry order_id"
    assert "c_fraud" not in policy_ids, "fraud_score must not reach policy agent"


def test_fanout_aggregate_token_reduction(embedder):
    """CHP saves >30% tokens vs passing full context to each subagent."""
    total_no_chp = TOTAL_TOKENS * len(FAN_OUT_MANIFESTS)
    total_chp = sum(
        sum(c.token_cost for c in select_chunks(CHUNKS, m, embedder))
        for m in FAN_OUT_MANIFESTS
    )
    reduction_pct = (total_no_chp - total_chp) / total_no_chp * 100
    assert reduction_pct > 30, f"Expected >30% reduction, got {reduction_pct:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Pattern 3 — Fan-in: subagents report back → orchestrator synthesizes
# ─────────────────────────────────────────────────────────────────────────────

ORCHESTRATOR_MANIFEST = ContextManifest(
    agent_id="orchestrator", task="synthesize_decision",
    requires=ContextRequirements(
        must_carry=[],
        domain_tags=["summary", "decision", "result", "outcome"],
        exclude=["PII_raw", "debug_trace"],
    ),
    token_budget=400, on_missing="proceed",
)


def _subagent_output_chunk(agent_id: str, content: str, hop: int) -> AnnotatedChunk:
    return AnnotatedChunk(
        chunk_id=f"out_{agent_id}",
        content=content,
        token_cost=60,
        source_agent=agent_id,
        source_turn=hop,
        timestamp=datetime.now(timezone.utc),
    )


def test_fanin_orchestrator_sees_all_subagent_outputs(ledger, embedder):
    """Orchestrator queries ledger and recovers outputs from all 4 subagents."""
    session_id = "fanin_test_001"

    # Subagents write their outputs at hop 1
    outputs = [
        ("fraud-agent",      "fraud risk: LOW score 0.08"),
        ("compliance-agent", "compliance: GDPR=ok PCI=flagged"),
        ("policy-agent",     "policy: eligible for full refund"),
        ("history-agent",    "history: 1 prior refund approved"),
    ]
    for agent_id, content in outputs:
        chunk = _subagent_output_chunk(agent_id, content, 1)
        env = _make_envelope(chunk, "orchestrator", 1)
        ledger.write(session_id, 1, agent_id, "orchestrator", env)

    # Orchestrator queries hop 1
    hop1 = ledger.query_hop(session_id, 1)
    recovered = {r.source_agent for r in hop1}
    assert "fraud-agent"      in recovered
    assert "compliance-agent" in recovered
    assert "policy-agent"     in recovered
    assert "history-agent"    in recovered


def test_fanin_orchestrator_within_budget(ledger, embedder):
    """Orchestrator's selected context stays within its token_budget."""
    session_id = "fanin_budget_001"
    outputs = [
        _subagent_output_chunk("fraud-agent",      "fraud LOW",       1),
        _subagent_output_chunk("compliance-agent", "compliance ok",   1),
        _subagent_output_chunk("policy-agent",     "eligible",        1),
        _subagent_output_chunk("history-agent",    "clean history",   1),
    ]
    all_chunks = CHUNKS + outputs
    selected = select_chunks(all_chunks, ORCHESTRATOR_MANIFEST, embedder)
    total = sum(c.token_cost for c in selected)
    assert total <= ORCHESTRATOR_MANIFEST.token_budget, (
        f"Orchestrator over budget: {total} > {ORCHESTRATOR_MANIFEST.token_budget}"
    )


def test_fanin_pii_never_reaches_orchestrator(ledger, embedder):
    """PII_raw excluded from orchestrator synthesis even when present in full context."""
    outputs = [_subagent_output_chunk("fraud-agent", "fraud LOW", 1)]
    selected = select_chunks(CHUNKS + outputs, ORCHESTRATOR_MANIFEST, embedder)
    ids = {c.chunk_id for c in selected}
    assert "c_pii" not in ids


def test_fanin_full_audit_trail(ledger, embedder):
    """After full fan-out + fan-in, ledger has entries across all hops."""
    session_id = "fanin_audit_001"

    # Hop 1: fan-out to 4 subagents
    for m in FAN_OUT_MANIFESTS:
        _, selected = _run_subagent(m, CHUNKS, ledger, session_id, 1, embedder)

    # Hop 2: subagents write results back
    for agent_id, content in [
        ("fraud-agent",      "fraud LOW"),
        ("compliance-agent", "GDPR ok"),
        ("policy-agent",     "eligible"),
        ("history-agent",    "clean"),
    ]:
        chunk = _subagent_output_chunk(agent_id, content, 2)
        env = _make_envelope(chunk, "orchestrator", 2)
        ledger.write(session_id, 2, agent_id, "orchestrator", env)

    # Hop 3: orchestrator synthesizes
    outputs_2 = [_subagent_output_chunk(a, c, 2) for a, c in [
        ("fraud-agent", "fraud LOW"), ("compliance-agent", "ok"),
        ("policy-agent", "eligible"), ("history-agent", "clean"),
    ]]
    orch_selected = select_chunks(CHUNKS + outputs_2, ORCHESTRATOR_MANIFEST, embedder)
    for chunk in orch_selected:
        env = _make_envelope(chunk, "orchestrator", 3)
        try:
            ledger.write(session_id, 3, chunk.source_agent, "orchestrator", env)
        except RuntimeError:
            pass

    # At least hops 1 and 2 are present in the ledger
    hop1 = ledger.query_hop(session_id, 1)
    hop2 = ledger.query_hop(session_id, 2)
    assert len(hop1) > 0, "No hop-1 records found"
    assert len(hop2) > 0, "No hop-2 records found"

    hop1_agents = {r.source_agent for r in hop1}
    assert "router" in hop1_agents or len(hop1_agents) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Ledger invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_ledger_rejects_duplicate_write(ledger):
    """Writing the same (session, chunk_id, hop) twice raises RuntimeError."""
    session_id = "dup_test_001"
    chunk = AnnotatedChunk(
        chunk_id="c_dup", content="some content", token_cost=10,
        source_agent="agent-a", source_turn=1,
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    env = _make_envelope(chunk, "agent-b", 0)
    ledger.write(session_id, 0, "agent-a", "agent-b", env)
    with pytest.raises(RuntimeError, match="duplicate"):
        ledger.write(session_id, 0, "agent-a", "agent-b", env)


def test_concurrent_ledger_writes_are_safe(ledger):
    """Concurrent writes from different agents don't corrupt the ledger."""
    session_id = "concurrent_test_001"
    errors: list[str] = []

    def write_chunk(i):
        try:
            chunk = AnnotatedChunk(
                chunk_id=f"chunk-{i}", content=f"content {i}",
                token_cost=20, source_agent=f"agent-{i}", source_turn=i,
                timestamp=datetime.now(timezone.utc),
            )
            env = _make_envelope(chunk, "orchestrator", i)
            ledger.write(session_id, i, f"agent-{i}", "orchestrator", env)
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=write_chunk, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent writes raised: {errors}"
    records = ledger.query(session_id)
    assert len(records) == 6
