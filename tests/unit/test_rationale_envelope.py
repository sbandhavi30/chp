import pytest
from chp.schema.rationale_envelope import RationaleEnvelope, AnnotatedChunk
from datetime import datetime, timezone


def test_annotated_chunk_construction():
    chunk = AnnotatedChunk(
        chunk_id="c_001",
        content="user_id is 42",
        token_cost=10,
        source_agent="auth-agent",
        source_turn=1,
        timestamp=datetime(2026, 8, 19, tzinfo=timezone.utc),
    )
    assert chunk.chunk_id == "c_001"
    assert chunk.token_cost == 10


def test_rationale_envelope_construction():
    env = RationaleEnvelope(
        chunk_id="c_047",
        content="order_id is 999",
        source_agent="auth-agent",
        source_turn=3,
        hop_sequence=["auth-agent", "orchestrator", "billing-agent"],
        selected_because=["domain_match:auth", "must_carry:order_id"],
        score=0.91,
        must_carry=True,
        token_cost=142,
        ledger_id="ledger_abc_turn_3",
    )
    assert env.score == 0.91
    assert "domain_match:auth" in env.selected_because
    assert env.must_carry is True


def test_rationale_envelope_score_bounds():
    with pytest.raises(Exception):
        RationaleEnvelope(
            chunk_id="c_x", content="x", source_agent="a", source_turn=0,
            hop_sequence=[], selected_because=[], score=1.5,
            must_carry=False, token_cost=1, ledger_id=None,
        )
