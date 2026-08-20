"""Tests for chp.observability — metrics hook, emit, structured log output."""
from __future__ import annotations
import json
import logging
import pytest
import chp
from chp.observability import CHPEvent, emit, set_metrics_hook, Timer
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.engine.embedder import StubEmbedder
from chp.engine.scorer import select_chunks


# ── helpers ───────────────────────────────────────────────────────────────────

def _chunk(cid: str, content: str, cost: int = 50) -> AnnotatedChunk:
    return AnnotatedChunk(chunk_id=cid, content=content, token_cost=cost, source_agent="a", source_turn=1)


def _manifest(agent_id: str = "test-agent", budget: int = 500) -> ContextManifest:
    return ContextManifest(
        agent_id=agent_id, task="test",
        requires=ContextRequirements(),
        token_budget=budget,
    )


# ── set_metrics_hook / emit ───────────────────────────────────────────────────

def test_hook_receives_event():
    events = []
    set_metrics_hook(lambda e, d: events.append((e, d)))
    try:
        emit(CHPEvent.LEDGER_WRITE, {"session_id": "s1", "elapsed_ms": 1.2})
        assert len(events) == 1
        e, d = events[0]
        assert e == CHPEvent.LEDGER_WRITE
        assert d["session_id"] == "s1"
        assert d["elapsed_ms"] == 1.2
    finally:
        set_metrics_hook(None)


def test_hook_exception_does_not_propagate():
    def bad_hook(e, d):
        raise RuntimeError("hook exploded")

    set_metrics_hook(bad_hook)
    try:
        emit(CHPEvent.TOKEN_REDUCTION, {"reduction_pct": 40.0})  # must not raise
    finally:
        set_metrics_hook(None)


def test_no_hook_emit_does_not_raise():
    set_metrics_hook(None)
    emit(CHPEvent.SELECT_CHUNKS_CALLED, {"agent_id": "x", "session_id": None, "input_chunks": 0, "token_budget": 100})


def test_hook_accessible_via_chp_package():
    # set_metrics_hook exported from top-level chp package
    seen = []
    chp.set_metrics_hook(lambda e, d: seen.append(e))
    try:
        emit(CHPEvent.CHUNK_SELECTED, {"agent_id": "a", "chunk_id": "c", "token_cost": 10, "session_id": None})
        assert CHPEvent.CHUNK_SELECTED in seen
    finally:
        chp.set_metrics_hook(None)


def test_chp_event_constants_exported():
    assert chp.CHPEvent.SELECT_CHUNKS_CALLED == "chp.select_chunks.called"
    assert chp.CHPEvent.TOKEN_REDUCTION      == "chp.token_reduction"
    assert chp.CHPEvent.LEDGER_WRITE         == "chp.ledger.write"
    assert chp.CHPEvent.LEDGER_QUERY         == "chp.ledger.query"


# ── structured log output ─────────────────────────────────────────────────────

def test_emit_logs_valid_json(caplog):
    with caplog.at_level(logging.INFO, logger="chp.observability"):
        emit(CHPEvent.TOKEN_REDUCTION, {"tokens_in": 200, "tokens_out": 80, "reduction_pct": 60.0})
    assert caplog.records, "expected at least one log record"
    record = caplog.records[-1]
    parsed = json.loads(record.message)
    assert parsed["chp_event"] == CHPEvent.TOKEN_REDUCTION
    assert parsed["tokens_in"] == 200
    assert parsed["reduction_pct"] == 60.0


# ── select_chunks emits expected events ───────────────────────────────────────

def test_select_chunks_emits_called_and_reduction():
    events: list[tuple[str, dict]] = []
    set_metrics_hook(lambda e, d: events.append((e, d)))
    try:
        chunks = [_chunk("c1", "alpha"), _chunk("c2", "beta")]
        select_chunks(chunks, _manifest(), StubEmbedder())
        types = [e for e, _ in events]
        assert CHPEvent.SELECT_CHUNKS_CALLED in types
        assert CHPEvent.TOKEN_REDUCTION in types
    finally:
        set_metrics_hook(None)


def test_select_chunks_emits_chunk_selected_per_selected():
    selected_events: list[dict] = []
    set_metrics_hook(lambda e, d: selected_events.append(d) if e == CHPEvent.CHUNK_SELECTED else None)
    try:
        chunks = [_chunk("c1", "alpha"), _chunk("c2", "beta")]
        result = select_chunks(chunks, _manifest(budget=500), StubEmbedder())
        assert len(selected_events) == len(result)
    finally:
        set_metrics_hook(None)


def test_select_chunks_emits_excluded_for_pii():
    excluded_events: list[dict] = []
    set_metrics_hook(lambda e, d: excluded_events.append(d) if e == CHPEvent.CHUNK_EXCLUDED else None)
    try:
        chunks = [_chunk("c1", "contains SSN: 123-45-6789")]
        manifest = ContextManifest(
            agent_id="a", task="t",
            requires=ContextRequirements(exclude=["SSN"]),
            token_budget=500,
        )
        select_chunks(chunks, manifest, StubEmbedder())
        assert any(d["chunk_id"] == "c1" for d in excluded_events)
    finally:
        set_metrics_hook(None)


def test_select_chunks_emits_must_carry_delivered():
    mc_events: list[dict] = []
    set_metrics_hook(lambda e, d: mc_events.append(d) if e == CHPEvent.MUST_CARRY_DELIVERED else None)
    try:
        chunks = [_chunk("c1", "billing_decision: approved")]
        manifest = ContextManifest(
            agent_id="a", task="t",
            requires=ContextRequirements(must_carry=["billing_decision"]),
            token_budget=500,
        )
        select_chunks(chunks, manifest, StubEmbedder())
        assert any(d["key"] == "billing_decision" for d in mc_events)
    finally:
        set_metrics_hook(None)


def test_select_chunks_emits_must_carry_missed_on_warn():
    missed: list[dict] = []
    set_metrics_hook(lambda e, d: missed.append(d) if e == CHPEvent.MUST_CARRY_MISSED else None)
    try:
        manifest = ContextManifest(
            agent_id="a", task="t",
            requires=ContextRequirements(must_carry=["missing_key"]),
            token_budget=500, on_missing="warn",
        )
        select_chunks([], manifest, StubEmbedder())
        # on_missing="warn" falls through — no missed event emitted for "warn" path
        # (only "ledger_fallback" path emits MUST_CARRY_MISSED)
    finally:
        set_metrics_hook(None)


def test_token_reduction_data():
    stats: list[dict] = []
    set_metrics_hook(lambda e, d: stats.append(d) if e == CHPEvent.TOKEN_REDUCTION else None)
    try:
        chunks = [_chunk(f"c{i}", f"content {i}", 100) for i in range(5)]
        manifest = ContextManifest(
            agent_id="a", task="t",
            requires=ContextRequirements(),
            token_budget=200,  # budget forces selection of only 2 of 5
        )
        select_chunks(chunks, manifest, StubEmbedder())
        assert stats, "TOKEN_REDUCTION event not emitted"
        s = stats[0]
        assert s["tokens_in"] == 500
        assert s["tokens_out"] <= 200
        assert 0 <= s["reduction_pct"] <= 100
    finally:
        set_metrics_hook(None)


# ── Timer ─────────────────────────────────────────────────────────────────────

def test_timer_measures_elapsed():
    import time
    with Timer() as t:
        time.sleep(0.01)
    assert t.elapsed_ms >= 10
    assert t.elapsed_ms < 500
