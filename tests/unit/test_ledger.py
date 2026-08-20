import pytest
from chp.ledger.lancedb_ledger import CHPLedger
from chp.schema.rationale_envelope import RationaleEnvelope


def _envelope(chunk_id="c_001", score=0.9, must_carry=False):
    return RationaleEnvelope(
        chunk_id=chunk_id,
        content=f"content of {chunk_id}",
        source_agent="agent-a",
        source_turn=1,
        hop_sequence=["agent-a", "agent-b"],
        selected_because=["domain_match:billing"],
        score=score,
        must_carry=must_carry,
        token_cost=50,
        ledger_id=None,
    )


def test_ledger_write_and_query():
    ledger = CHPLedger()
    env = _envelope("c_001")
    ledger_id = ledger.write("session_1", hop_number=0, from_agent="a", to_agent="b", envelope=env)
    assert ledger_id is not None
    results = ledger.query("session_1")
    assert len(results) == 1
    assert results[0].chunk_id == "c_001"


def test_ledger_append_only_rejects_duplicate():
    ledger = CHPLedger()
    env = _envelope("c_dup")
    ledger.write("session_1", hop_number=0, from_agent="a", to_agent="b", envelope=env)
    with pytest.raises(RuntimeError, match="duplicate"):
        ledger.write("session_1", hop_number=0, from_agent="a", to_agent="b", envelope=env)


def test_ledger_query_by_hop():
    ledger = CHPLedger()
    ledger.write("s1", hop_number=0, from_agent="a", to_agent="b", envelope=_envelope("c_hop0"))
    ledger.write("s1", hop_number=1, from_agent="b", to_agent="c", envelope=_envelope("c_hop1"))
    hop0 = ledger.query_hop("s1", hop_number=0)
    assert len(hop0) == 1
    assert hop0[0].chunk_id == "c_hop0"


def test_ledger_query_isolated_by_session():
    ledger = CHPLedger()
    ledger.write("sess_x", hop_number=0, from_agent="a", to_agent="b", envelope=_envelope("c_x"))
    ledger.write("sess_y", hop_number=0, from_agent="a", to_agent="b", envelope=_envelope("c_y"))
    results = ledger.query("sess_x")
    ids = [r.chunk_id for r in results]
    assert "c_x" in ids
    assert "c_y" not in ids
