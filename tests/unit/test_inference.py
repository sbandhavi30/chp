from chp.inference import infer_manifest
from chp.schema.context_manifest import ContextManifest


def test_infer_billing_heuristic():
    m = infer_manifest(
        role="Billing Specialist",
        goal="Resolve duplicate charge disputes and process refunds",
        backstory="You handle billing issues for premium customers",
    )
    assert isinstance(m, ContextManifest)
    assert "order_id" in m.requires.must_carry
    assert "user_id" in m.requires.must_carry
    assert any("billing" in tag for tag in m.requires.domain_tags)
    assert "PII_raw" in m.requires.exclude


def test_infer_auth_heuristic():
    m = infer_manifest(
        role="Auth Specialist",
        goal="Verify customer identity and authentication status",
    )
    assert "user_id" in m.requires.must_carry
    assert any("auth" in tag for tag in m.requires.domain_tags)


def test_infer_fraud_heuristic():
    m = infer_manifest(
        role="Fraud Analyst",
        goal="Assess fraud risk and flag suspicious transactions",
    )
    assert "fraud_score" in m.requires.must_carry
    assert any("fraud" in tag for tag in m.requires.domain_tags)


def test_infer_compliance_keeps_pii():
    m = infer_manifest(
        role="Compliance Officer",
        goal="Check GDPR and PCI compliance for customer data handling",
    )
    # compliance agent needs PII flags — should not be excluded
    assert "PII_raw" not in m.requires.exclude


def test_infer_unknown_agent_returns_generic():
    m = infer_manifest(
        role="Quantum Entanglement Resolver",
        goal="Resolve entanglement issues in the quantum pipeline",
    )
    assert isinstance(m, ContextManifest)
    assert m.token_budget > 0
    assert m.agent_id == "quantum_entanglement_resolver"


def test_infer_agent_id_slugified():
    m = infer_manifest(
        role="Senior Billing & Refund Specialist",
        goal="Handle refunds",
    )
    assert " " not in m.agent_id
    assert "&" not in m.agent_id


def test_infer_custom_agent_id():
    m = infer_manifest(
        role="Billing Specialist",
        goal="Handle refunds",
        agent_id="my-custom-billing-v2",
    )
    assert m.agent_id == "my-custom-billing-v2"


# ── accept_upstream_output heuristic inference ────────────────────────────────

def test_router_accept_upstream_false():
    """Router is first-hop — no upstream decisions exist."""
    m = infer_manifest(role="Support Router", goal="Classify and dispatch incoming tickets")
    assert m.requires.accept_upstream_output is False


def test_auth_accept_upstream_false():
    """Auth is early-hop — works from pool context only."""
    m = infer_manifest(role="Auth Specialist", goal="Verify customer identity")
    assert m.requires.accept_upstream_output is False


def test_orchestrator_accept_upstream_true():
    """Orchestrator synthesizes ALL specialist results — needs everyone's output."""
    m = infer_manifest(
        role="Orchestrator",
        goal="Synthesize results from all specialist subagents and produce final decision",
    )
    assert m.requires.accept_upstream_output is True


def test_fraud_accept_upstream_billing_without_pipeline():
    """Without pipeline_agents, fraud gets True (upstream expected) not a named list."""
    m = infer_manifest(
        role="Fraud Analyst",
        goal="Assess fraud risk based on billing and velocity signals",
    )
    # No pipeline_agents provided — heuristic returns True (upstream expected)
    assert m.requires.accept_upstream_output is not False


def test_fraud_accept_upstream_named_with_pipeline():
    """With pipeline_agents, fraud gets exact agent ID that matches 'billing' keyword."""
    m = infer_manifest(
        role="Fraud Analyst",
        goal="Assess fraud risk based on billing and velocity signals",
        pipeline_agents=["router", "auth-agent", "billing-agent", "compliance", "policy"],
    )
    accept = m.requires.accept_upstream_output
    assert isinstance(accept, list), f"Expected list, got {accept}"
    assert any("billing" in a for a in accept), \
        f"billing-agent must be in accept list: {accept}"
    assert "router" not in accept, "router must not be in fraud's upstream deps"
    assert "compliance" not in accept, "compliance must not be in fraud's upstream deps"


def test_summarizer_accept_upstream_named_with_pipeline():
    """Summarizer needs orchestrator and escalation output specifically."""
    m = infer_manifest(
        role="Customer Summarizer",
        goal="Summarize the resolution outcome for the customer-facing response",
        pipeline_agents=["router", "billing-agent", "orchestrator-agent", "escalation-manager"],
    )
    accept = m.requires.accept_upstream_output
    assert isinstance(accept, list), f"Expected list, got {accept}"
    assert any("orchestrat" in a for a in accept), \
        f"orchestrator must be in summarizer's upstream: {accept}"


def test_billing_accept_upstream_auth_with_pipeline():
    """Billing needs auth confirmation before approving refund."""
    m = infer_manifest(
        role="Billing Specialist",
        goal="Resolve duplicate charge disputes and process refunds",
        pipeline_agents=["router", "auth-specialist", "fraud-agent", "compliance"],
    )
    accept = m.requires.accept_upstream_output
    assert isinstance(accept, list), f"Expected list, got {accept}"
    assert any("auth" in a for a in accept), \
        f"auth must be in billing's upstream deps: {accept}"


def test_research_accept_upstream_false():
    """Research is standalone — works from pool context only."""
    m = infer_manifest(
        role="Research Agent",
        goal="Search and retrieve relevant knowledge base articles",
        pipeline_agents=["router", "billing-agent", "orchestrator"],
    )
    assert m.requires.accept_upstream_output is False


def test_pipeline_agents_filters_nonexistent_ids():
    """LLM returning agent IDs not in pipeline_agents are filtered out."""
    from unittest.mock import MagicMock
    import json as _json
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = _json.dumps({
        "must_carry": ["fraud_score"],
        "domain_tags": ["fraud"],
        "history_depth": "decisions_only",
        "exclude": ["PII_raw"],
        "token_budget": 600,
        "on_missing": "warn",
        "accept_upstream_output": ["billing-agent", "nonexistent-agent-xyz"],
        "reasoning": "test",
    })
    client.chat.completions.create.return_value = MagicMock(choices=[choice])

    m = infer_manifest(
        role="Fraud Analyst",
        goal="Assess fraud risk",
        pipeline_agents=["router", "billing-agent", "auth"],
        llm_client=client,
    )
    accept = m.requires.accept_upstream_output
    assert "billing-agent" in accept
    assert "nonexistent-agent-xyz" not in accept, \
        "IDs not in pipeline_agents must be filtered out"


def test_accept_upstream_false_when_llm_returns_empty_list():
    """LLM returning [] for accept_upstream_output → False (safe default)."""
    from unittest.mock import MagicMock
    import json as _json
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = _json.dumps({
        "must_carry": [], "domain_tags": ["x"],
        "history_depth": "full", "exclude": [],
        "token_budget": 500, "on_missing": "warn",
        "accept_upstream_output": [],
        "reasoning": "test",
    })
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    m = infer_manifest(role="Some Agent", goal="Do stuff", llm_client=client)
    assert m.requires.accept_upstream_output is False
