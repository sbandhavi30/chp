from datetime import datetime, timezone
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.engine.embedder import StubEmbedder
from chp.ledger.lancedb_ledger import CHPLedger
from chp.adapters.autogen import CHPConversableAgent


def test_chp_autogen_agent_filters_chunks():
    manifest = ContextManifest(
        chp_version="0.1", agent_id="summarizer", task="summarize_ticket",
        requires=ContextRequirements(
            must_carry=["ticket_id"], domain_tags=["support"],
            history_depth="full", recency_window=None, exclude=[],
        ),
        token_budget=500, on_missing="warn",
    )
    chunks = [
        AnnotatedChunk(chunk_id="c1", content="ticket_id is T-42", token_cost=40,
                       source_agent="router", source_turn=1,
                       timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
        AnnotatedChunk(chunk_id="c2", content="random noise", token_cost=40,
                       source_agent="router", source_turn=2,
                       timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc)),
    ]

    agent = CHPConversableAgent(
        manifest=manifest, ledger=CHPLedger(), embedder=StubEmbedder(), name="summarizer",
    )
    selected = agent.select_context(chunks=chunks, session_id="sess_ag", hop=0)
    assert "c1" in [c.chunk_id for c in selected]
