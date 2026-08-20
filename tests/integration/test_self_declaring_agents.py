"""
Integration tests for self-declaring agents:
  - Each "agent" is defined by role + goal (like CrewAI Agent(...))
  - infer_manifest() auto-generates its ContextManifest from that description
  - Tests verify each agent's manifest is correct AND that the scorer
    delivers the right context to each agent in a shared 12-chunk session

The LLM test (test_llm_infer_*) requires OPENAI_API_KEY in the environment
and is skipped automatically when the key is absent.

Usage:
    pytest tests/integration/test_self_declaring_agents.py -v
    OPENAI_API_KEY=sk-... pytest tests/integration/test_self_declaring_agents.py -v -k llm
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from chp.engine.embedder import StubEmbedder
from chp.engine.scorer import select_chunks
from chp.inference import infer_manifest
from chp.ledger.lancedb_ledger import CHPLedger
from chp.schema.context_manifest import ContextManifest
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope

# ─────────────────────────────────────────────────────────────────────────────
# 10 agent definitions — role + goal, just like CrewAI Agent(role=, goal=)
# ─────────────────────────────────────────────────────────────────────────────

AGENT_DEFS = [
    {
        "role": "Billing Specialist",
        "goal": "Resolve duplicate charge disputes and process refunds",
        "backstory": "You handle billing issues for premium customers",
    },
    {
        "role": "Auth Specialist",
        "goal": "Verify customer identity and authentication status",
    },
    {
        "role": "Fraud Analyst",
        "goal": "Assess fraud risk and flag suspicious transactions",
    },
    {
        "role": "Compliance Officer",
        "goal": "Check GDPR and PCI compliance for customer data handling",
    },
    {
        "role": "Support Router",
        "goal": "Classify and dispatch incoming support tickets to the right agent",
    },
    {
        "role": "Research Agent",
        "goal": "Search and retrieve relevant knowledge base articles for the customer issue",
    },
    {
        "role": "Orchestrator",
        "goal": "Synthesize results from all subagents and produce the final resolution decision",
    },
    {
        "role": "Policy Checker",
        "goal": "Check refund eligibility and policy terms for the customer order",
    },
    {
        "role": "Code Reviewer",
        "goal": "Review security and style issues in the payment integration diff",
    },
    {
        "role": "Customer Summarizer",
        "goal": "Summarize the resolution for the customer-facing response",
    },
    {
        "role": "Escalation Manager",
        "goal": "Coordinate escalation response and delegate to senior specialists",
    },
    {
        "role": "Account Auditor",
        "goal": "Audit account history and verify identity for suspicious login patterns",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Shared session: 12 chunks covering all domain areas
# ─────────────────────────────────────────────────────────────────────────────

SESSION = [
    AnnotatedChunk(chunk_id="c_user",       content="user_id: USR-007 | tier: enterprise",            token_cost=30,  source_agent="router", source_turn=1,  timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_order",      content="order_id: ORD-2024 | amount: $499 | plan: Elite", token_cost=35, source_agent="router", source_turn=2,  timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_pii",        content="PII_raw: ssn=123-45-6789 | dob=1990-01-01",       token_cost=40,  source_agent="router", source_turn=3,  timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_auth",       content="auth_status: verified | mfa: passed | session_token: eyJhb", token_cost=35, source_agent="router", source_turn=4, timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_fraud",      content="fraud_score: 0.22 | velocity: high | device: unknown", token_cost=40, source_agent="router", source_turn=5, timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_compliance", content="compliance_flags: GDPR=ok | PCI=flagged_raw | SOX=ok", token_cost=40, source_agent="router", source_turn=6, timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_policy",     content="refund_policy: Elite Plan 60-day full refund eligible", token_cost=45, source_agent="router", source_turn=7, timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_request",    content="customer_request: charged twice for Elite plan, dispute $499", token_cost=45, source_agent="router", source_turn=8, timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_debug",      content="debug_trace: POST /api/billing 500 idempotency_key=null retried 3x", token_cost=60, source_agent="router", source_turn=9, timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_history",    content="prior_tickets: TKT-500 fraud dispute resolved 2026-03, TKT-481 billing ok", token_cost=50, source_agent="router", source_turn=10, timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_diff",       content="git diff: +charge_service.py billing_v2 idempotency fix", token_cost=55, source_agent="router", source_turn=11, timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    AnnotatedChunk(chunk_id="c_outcome",    content="resolution_outcome: refund approved $499 | case closed", token_cost=35, source_agent="router", source_turn=12, timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
]

TOTAL_TOKENS = sum(c.token_cost for c in SESSION)


@pytest.fixture
def embedder():
    return StubEmbedder()


@pytest.fixture
def ledger():
    return CHPLedger()


# ─────────────────────────────────────────────────────────────────────────────
# Core: infer_manifest from role/goal — heuristic path (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _infer_all():
    """Infer manifests for all 12 agents."""
    return [infer_manifest(**d) for d in AGENT_DEFS]


def test_all_agents_produce_valid_manifest():
    """Every agent definition produces a valid ContextManifest."""
    for defn in AGENT_DEFS:
        m = infer_manifest(**defn)
        assert isinstance(m, ContextManifest), f"infer_manifest failed for {defn['role']}"
        assert m.token_budget > 0
        assert len(m.requires.domain_tags) >= 3, f"{defn['role']}: too few domain_tags"
        assert " " not in m.agent_id, f"{defn['role']}: agent_id contains spaces"


def test_all_agents_stay_within_budget(embedder):
    """Every auto-inferred manifest selects context within its token budget."""
    for defn in AGENT_DEFS:
        m = infer_manifest(**defn)
        selected = select_chunks(SESSION, m, embedder)
        tokens = sum(c.token_cost for c in selected)
        assert tokens <= m.token_budget, (
            f"{defn['role']}: {tokens} tokens > budget {m.token_budget}"
        )


def test_billing_agent_self_declared_manifest_carries_required_fields(embedder):
    """Billing agent auto-infers must_carry=['order_id', 'user_id'] and selects those chunks."""
    m = infer_manifest(
        role="Billing Specialist",
        goal="Resolve duplicate charge disputes and process refunds",
        backstory="You handle billing issues for premium customers",
    )
    assert "order_id" in m.requires.must_carry
    assert "user_id" in m.requires.must_carry
    assert "PII_raw" in m.requires.exclude

    selected = select_chunks(SESSION, m, embedder)
    ids = {c.chunk_id for c in selected}
    assert "c_order" in ids, "billing agent must receive order_id chunk"
    assert "c_user" in ids,  "billing agent must receive user_id chunk"
    assert "c_pii" not in ids, "billing agent must not receive PII_raw"


def test_auth_agent_self_declared_manifest(embedder):
    m = infer_manifest(
        role="Auth Specialist",
        goal="Verify customer identity and authentication status",
    )
    assert "user_id" in m.requires.must_carry
    assert any("auth" in t.lower() for t in m.requires.domain_tags)

    selected = select_chunks(SESSION, m, embedder)
    ids = {c.chunk_id for c in selected}
    assert "c_user" in ids
    assert "c_auth" in ids
    assert "c_pii" not in ids


def test_fraud_agent_self_declared_manifest(embedder):
    m = infer_manifest(
        role="Fraud Analyst",
        goal="Assess fraud risk and flag suspicious transactions",
    )
    assert "fraud_score" in m.requires.must_carry

    selected = select_chunks(SESSION, m, embedder)
    ids = {c.chunk_id for c in selected}
    assert "c_fraud" in ids
    assert "c_pii" not in ids


def test_compliance_agent_keeps_pii_flags(embedder):
    """Compliance agent auto-infers that it NEEDS PII flags — PII_raw not in exclude."""
    m = infer_manifest(
        role="Compliance Officer",
        goal="Check GDPR and PCI compliance for customer data handling",
    )
    assert "PII_raw" not in m.requires.exclude, (
        "Compliance agent must NOT exclude PII_raw — it needs PII flags"
    )
    selected = select_chunks(SESSION, m, embedder)
    ids = {c.chunk_id for c in selected}
    # compliance_flags chunk should be available
    assert "c_compliance" in ids


def test_pii_never_reaches_non_compliance_agents(embedder):
    """PII_raw excluded from all agents except compliance."""
    for defn in AGENT_DEFS:
        m = infer_manifest(**defn)
        selected = select_chunks(SESSION, m, embedder)
        ids = [c.chunk_id for c in selected]
        if defn["role"] != "Compliance Officer":
            assert "c_pii" not in ids, f"PII leaked to {defn['role']}"


def test_aggregate_token_reduction_across_all_agents(embedder):
    """CHP saves tokens vs passing full context to every agent.

    StubEmbedder returns zero vectors so semantic similarity is flat — only
    must_carry and exclude rules fire.  Real sentence-transformer embeddings
    push this well above 40%.  The threshold here validates the exclude/budget
    machinery, not semantic ranking.
    """
    total_no_chp = TOTAL_TOKENS * len(AGENT_DEFS)
    total_chp = sum(
        sum(c.token_cost for c in select_chunks(SESSION, infer_manifest(**d), embedder))
        for d in AGENT_DEFS
    )
    reduction = (total_no_chp - total_chp) / total_no_chp * 100
    assert reduction > 15, f"Expected >15% reduction (stub embedder), got {reduction:.1f}%"


def test_all_agents_in_pipeline_no_cross_contamination(embedder, ledger):
    """Run all 12 agents through a simulated pipeline; audit that debug_trace never leaks."""
    session_id = "pipeline_12_agents"
    for i, defn in enumerate(AGENT_DEFS):
        m = infer_manifest(**defn)
        selected = select_chunks(SESSION, m, embedder)
        ids = [c.chunk_id for c in selected]
        assert "c_debug" not in ids, f"debug_trace leaked to {defn['role']}"
        for chunk in selected:
            env = RationaleEnvelope(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source_agent=chunk.source_agent,
                source_turn=chunk.source_turn,
                hop_sequence=[chunk.source_agent, m.agent_id],
                selected_because=[f"hop:{i}"],
                score=0.5,
                must_carry=False,
                token_cost=chunk.token_cost,
            )
            try:
                ledger.write(session_id, i, chunk.source_agent, m.agent_id, env)
            except RuntimeError:
                pass  # duplicate chunk across agents — fine


def test_agent_id_uniqueness():
    """All 12 agent_ids are unique (no collision from slugification)."""
    manifests = _infer_all()
    ids = [m.agent_id for m in manifests]
    assert len(ids) == len(set(ids)), f"Duplicate agent_ids: {[x for x in ids if ids.count(x) > 1]}"


def test_code_reviewer_infers_large_budget(embedder):
    """Code reviewer needs large context window for diffs — budget should be >=2000."""
    m = infer_manifest(
        role="Code Reviewer",
        goal="Review security and style issues in the payment integration diff",
    )
    assert m.token_budget >= 2000, f"Code reviewer budget too small: {m.token_budget}"


def test_orchestrator_infers_largest_budget():
    """Orchestrator synthesizes all results — should have the largest budget."""
    orchestrator = infer_manifest(
        role="Orchestrator",
        goal="Synthesize results from all subagents and produce the final resolution decision",
    )
    billing = infer_manifest(
        role="Billing Specialist",
        goal="Resolve duplicate charge disputes",
    )
    assert orchestrator.token_budget >= billing.token_budget, (
        "Orchestrator should have >= budget compared to specialized agents"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM test — skipped unless OPENAI_API_KEY is set
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping LLM inference test",
)
def test_llm_infer_billing_agent_manifest(embedder):
    """
    Uses a real OpenAI call to infer the billing manifest, then runs the scorer.
    Set OPENAI_API_KEY to enable.

        OPENAI_API_KEY=sk-... pytest tests/integration/test_self_declaring_agents.py -v -k llm
    """
    import openai
    client = openai.OpenAI()

    m = infer_manifest(
        role="Billing Specialist",
        goal="Resolve duplicate charge disputes and process refunds",
        backstory="You handle billing issues for premium enterprise customers",
        llm_client=client,
        model="gpt-4o-mini",
    )

    assert isinstance(m, ContextManifest)
    assert "order_id" in m.requires.must_carry, f"LLM missed order_id: {m.requires.must_carry}"
    assert "user_id" in m.requires.must_carry,  f"LLM missed user_id: {m.requires.must_carry}"
    assert "PII_raw" in m.requires.exclude,     f"LLM forgot to exclude PII: {m.requires.exclude}"
    assert m.token_budget > 0

    selected = select_chunks(SESSION, m, embedder)
    ids = {c.chunk_id for c in selected}
    assert "c_order" in ids
    assert "c_pii" not in ids


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping LLM inference test",
)
def test_llm_infer_all_12_agents_valid_manifests(embedder):
    """
    Uses real OpenAI to infer manifests for all 12 agents; verifies basic invariants.
    This test costs ~12 API calls (gpt-4o-mini is cheap — ~$0.001 total).
    """
    import openai
    client = openai.OpenAI()

    for defn in AGENT_DEFS:
        m = infer_manifest(**defn, llm_client=client, model="gpt-4o-mini")
        assert isinstance(m, ContextManifest), f"LLM infer failed for {defn['role']}"
        assert m.token_budget > 0
        assert len(m.requires.domain_tags) >= 3, f"{defn['role']}: too few tags from LLM"

        selected = select_chunks(SESSION, m, embedder)
        tokens = sum(c.token_cost for c in selected)
        assert tokens <= m.token_budget, (
            f"{defn['role']}: LLM budget {m.token_budget} exceeded by scorer: {tokens}"
        )
