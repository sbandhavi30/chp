from datetime import datetime, timezone
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.engine.embedder import StubEmbedder
from chp.ledger.lancedb_ledger import CHPLedger
from chp.adapters.langgraph import chp_node_middleware


def test_langgraph_middleware_filters_context():
    manifest = ContextManifest(
        chp_version="0.1",
        agent_id="billing-agent",
        task="resolve_refund",
        requires=ContextRequirements(
            must_carry=["order_id"],
            domain_tags=["billing"],
            history_depth="full",
            recency_window=None,
            exclude=["debug_trace"],
        ),
        token_budget=200,
        on_missing="warn",
    )
    ledger = CHPLedger()
    embedder = StubEmbedder()

    chunks = [
        AnnotatedChunk(chunk_id="c1", content="order_id is 42", token_cost=20,
                       source_agent="auth", source_turn=1,
                       timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
        AnnotatedChunk(chunk_id="c2", content="debug_trace: stack overflow", token_cost=20,
                       source_agent="auth", source_turn=2,
                       timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
        AnnotatedChunk(chunk_id="c3", content="billing amount $99", token_cost=20,
                       source_agent="auth", source_turn=3,
                       timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    ]

    state = {"chp_chunks": chunks, "chp_session_id": "sess_test", "chp_hop": 0}

    @chp_node_middleware(manifest=manifest, ledger=ledger, embedder=embedder)
    def billing_node(state):
        return {"injected_chunks": state["chp_selected_chunks"]}

    result = billing_node(state)
    selected_ids = [c.chunk_id for c in result["injected_chunks"]]
    assert "c1" in selected_ids
    assert "c2" not in selected_ids


def test_langgraph_middleware_writes_to_ledger():
    manifest = ContextManifest(
        chp_version="0.1",
        agent_id="summarizer",
        task="summarize",
        requires=ContextRequirements(
            must_carry=[], domain_tags=["summary"],
            history_depth="full", recency_window=None, exclude=[],
        ),
        token_budget=500,
        on_missing="warn",
    )
    ledger = CHPLedger()
    chunks = [
        AnnotatedChunk(chunk_id="cx1", content="fact one", token_cost=30,
                       source_agent="agent-a", source_turn=1,
                       timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    ]
    state = {"chp_chunks": chunks, "chp_session_id": "sess_ledger", "chp_hop": 0}

    @chp_node_middleware(manifest=manifest, ledger=ledger, embedder=StubEmbedder())
    def node(state):
        return state

    node(state)
    results = ledger.query("sess_ledger")
    assert len(results) >= 1
