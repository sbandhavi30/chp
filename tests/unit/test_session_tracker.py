"""
Unit tests for SessionTokenTracker — per-session token accumulation across hops.
"""
from __future__ import annotations

import chp
from chp.observability import SessionTokenTracker, CHPEvent
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.engine.scorer import select_chunks
from chp.engine.embedder import StubEmbedder


def _chunk(cid, cost):
    return AnnotatedChunk(
        chunk_id=cid, content=f"content {cid} billing payment",
        token_cost=cost, source_agent="src", source_turn=0,
    )


def _manifest(budget=10000):
    return ContextManifest(
        agent_id="test-agent",
        task="test",
        requires=ContextRequirements(),
        token_budget=budget,
    )


# ── Basic accumulation ────────────────────────────────────────────────────────

class TestSessionTokenTracker:
    def setup_method(self):
        chp.set_metrics_hook(None)

    def teardown_method(self):
        chp.set_metrics_hook(None)

    def test_accumulates_across_hops(self):
        tracker = SessionTokenTracker("s1")
        chp.set_metrics_hook(tracker.on_event)

        for _ in range(3):
            select_chunks(
                [_chunk("a", 100), _chunk("b", 200)],
                _manifest(),
                StubEmbedder(),
                session_id="s1",
            )

        summary = tracker.close()
        assert summary["hops"] == 3
        assert summary["total_tokens_in"] == 3 * 300
        assert summary["total_tokens_out"] <= summary["total_tokens_in"]
        assert summary["session_id"] == "s1"

    def test_reduction_pct_correct(self):
        tracker = SessionTokenTracker("s2")
        chp.set_metrics_hook(tracker.on_event)

        # budget=50 forces only first chunk (cost 40) selected out of 40+200
        select_chunks(
            [_chunk("small", 40), _chunk("big", 200)],
            _manifest(budget=50),
            StubEmbedder(),
            session_id="s2",
        )
        summary = tracker.close()
        assert summary["total_tokens_in"] == 240
        assert summary["total_tokens_out"] == 40
        assert summary["overall_reduction_pct"] == round((1 - 40/240) * 100, 1)

    def test_close_emits_session_summary_event(self):
        events = []
        tracker = SessionTokenTracker("s3")
        chp.set_metrics_hook(tracker.on_event)

        select_chunks([_chunk("x", 50)], _manifest(), StubEmbedder(), session_id="s3")

        # install second hook to catch SESSION_SUMMARY
        chp.set_metrics_hook(lambda e, d: events.append((e, d)))
        summary = tracker.close()

        session_events = [e for e in events if e[0] == SessionTokenTracker.SESSION_SUMMARY]
        assert len(session_events) == 1
        assert session_events[0][1]["session_id"] == "s3"

    def test_ignores_other_session_events(self):
        tracker = SessionTokenTracker("session-A")
        chp.set_metrics_hook(tracker.on_event)

        # fire TOKEN_REDUCTION for a different session_id
        select_chunks(
            [_chunk("y", 100)],
            _manifest(),
            StubEmbedder(),
            session_id="session-B",
        )
        summary = tracker.close()
        assert summary["hops"] == 0
        assert summary["total_tokens_in"] == 0

    def test_no_session_id_counts_all(self):
        # When session_id is None in TOKEN_REDUCTION, tracker accepts it
        tracker = SessionTokenTracker("any")
        chp.set_metrics_hook(tracker.on_event)

        # select_chunks with no session_id → emits TOKEN_REDUCTION with session_id=None
        select_chunks([_chunk("z", 80)], _manifest(), StubEmbedder())
        summary = tracker.close()
        assert summary["hops"] == 1

    def test_reset_clears_counters(self):
        tracker = SessionTokenTracker("s4")
        chp.set_metrics_hook(tracker.on_event)

        select_chunks([_chunk("a", 100)], _manifest(), StubEmbedder(), session_id="s4")
        tracker.reset()
        summary = tracker.close()
        assert summary["hops"] == 0
        assert summary["total_tokens_in"] == 0

    def test_upstream_hook_called(self):
        upstream_events = []
        tracker = SessionTokenTracker(
            "s5",
            upstream_hook=lambda e, d: upstream_events.append(e),
        )
        chp.set_metrics_hook(tracker.on_event)

        select_chunks([_chunk("a", 50)], _manifest(), StubEmbedder(), session_id="s5")
        tracker.close()

        assert CHPEvent.TOKEN_REDUCTION in upstream_events

    def test_upstream_hook_exception_does_not_crash(self):
        def bad_hook(e, d):
            raise RuntimeError("upstream exploded")

        tracker = SessionTokenTracker("s6", upstream_hook=bad_hook)
        chp.set_metrics_hook(tracker.on_event)

        # Should not raise
        select_chunks([_chunk("a", 50)], _manifest(), StubEmbedder(), session_id="s6")
        summary = tracker.close()
        assert summary["hops"] == 1

    def test_zero_tokens_in_no_div_zero(self):
        tracker = SessionTokenTracker("empty")
        summary = tracker.close()
        assert summary["overall_reduction_pct"] == 0.0

    def test_summary_keys_present(self):
        tracker = SessionTokenTracker("s7")
        summary = tracker.close()
        for key in ("session_id", "hops", "total_tokens_in", "total_tokens_out", "overall_reduction_pct"):
            assert key in summary


# ── Integration: benchmark pipeline produces correct session summary ──────────

def test_benchmark_pipeline_session_summary():
    from chp.benchmarks.compare import _make_chunk_pool, _billing_manifest

    manifest = _billing_manifest()
    embedder = StubEmbedder()
    tracker = SessionTokenTracker("bench-s")
    chp.set_metrics_hook(tracker.on_event)

    for hop in range(3):
        select_chunks(_make_chunk_pool(hop), manifest, embedder, session_id="bench-s")

    summary = tracker.close()
    chp.set_metrics_hook(None)

    assert summary["hops"] == 3
    assert summary["total_tokens_in"] > summary["total_tokens_out"]
    assert 0 < summary["overall_reduction_pct"] < 100
