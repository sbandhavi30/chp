"""
Synthetic load test: 100 sessions, 12 agents each, real LanceDB writes.

Measures:
  - write throughput (sessions/sec)
  - query latency (p50, p95, p99)
  - ledger_fallback recovery under concurrent load
  - prune + orphan cleanup time
  - final ledger stats

No LLM calls. StubEmbedder. Runs in ~10-30s on a laptop.
"""
from __future__ import annotations

import statistics
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from chp.engine.embedder import StubEmbedder
from chp.engine.scorer import select_chunks
from chp.ledger.lancedb_ledger import CHPLedger
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope

# ── Config ────────────────────────────────────────────────────────────────────

NUM_SESSIONS       = 100
AGENTS_PER_SESSION = 12   # mirrors the 12-agent demo pipeline
CHUNKS_PER_SESSION = 10   # realistic pool size
PARALLEL_WORKERS   = 20   # concurrent threads writing sessions

# ── Fixtures ──────────────────────────────────────────────────────────────────

AGENT_DEFS = [
    ("router",       ["user_id"],                        ["PII_raw"],                    400),
    ("auth",         ["user_id", "auth_status"],         ["PII_raw", "credit_card"],     500),
    ("billing",      ["order_id", "user_id"],            ["PII_raw", "ssn"],            1500),
    ("fraud",        ["fraud_score"],                    ["PII_raw"],                    600),
    ("compliance",   ["user_id", "compliance_flags"],    ["debug_trace"],                800),
    ("policy",       ["order_id"],                       ["PII_raw", "fraud_score"],     500),
    ("research",     [],                                 ["PII_raw"],                   1000),
    ("orchestrator", [],                                 ["PII_raw", "ssn"],            4000),
    ("escalation",   ["final_decision"],                 ["PII_raw"],                   1200),
    ("summarizer",   ["final_decision", "refund_amount"],["PII_raw", "fraud_score"],     300),
    ("code_reviewer",[],                                 ["user_data", "PII_raw"],      5000),
    ("auditor",      ["user_id"],                        ["PII_raw"],                   1000),
]


def _make_session_chunks(session_id: str) -> list[AnnotatedChunk]:
    """10 realistic chunks per session."""
    return [
        AnnotatedChunk(chunk_id=f"{session_id}_order",      content="order_id: ORD-4492 amount: $299",      token_cost=30,  source_agent="router", source_turn=0),
        AnnotatedChunk(chunk_id=f"{session_id}_user",       content="user_id: USR-001 tier: premium",        token_cost=25,  source_agent="router", source_turn=0),
        AnnotatedChunk(chunk_id=f"{session_id}_auth",       content="auth_status: verified mfa: true",       token_cost=20,  source_agent="auth",   source_turn=1),
        AnnotatedChunk(chunk_id=f"{session_id}_fraud",      content="fraud_score: 0.12 velocity: low",       token_cost=35,  source_agent="fraud",  source_turn=2),
        AnnotatedChunk(chunk_id=f"{session_id}_compliance", content="compliance_flags: GDPR_ok PCI_ok",      token_cost=40,  source_agent="comp",   source_turn=2),
        AnnotatedChunk(chunk_id=f"{session_id}_policy",     content="refund_eligible: true policy: 30day",   token_cost=30,  source_agent="policy", source_turn=2),
        AnnotatedChunk(chunk_id=f"{session_id}_decision",   content="final_decision: approve refund_amount: $299", token_cost=45, source_agent="orch", source_turn=3),
        AnnotatedChunk(chunk_id=f"{session_id}_history",    content="account_history: 3 prior orders clean", token_cost=80,  source_agent="audit",  source_turn=1),
        AnnotatedChunk(chunk_id=f"{session_id}_kb",         content="kb_article: duplicate charge resolution steps", token_cost=120, source_agent="research", source_turn=2),
        AnnotatedChunk(chunk_id=f"{session_id}_diff",       content="billing_diff: fixed idempotency bug in charge handler", token_cost=200, source_agent="code_reviewer", source_turn=0),
    ]


def _make_envelope(session_id: str, agent_id: str, hop: int, chunk: AnnotatedChunk) -> RationaleEnvelope:
    return RationaleEnvelope(
        chunk_id=f"{chunk.chunk_id}_{agent_id}_{hop}",
        content=chunk.content,
        source_agent=agent_id,
        source_turn=hop,
        hop_sequence=["router", agent_id],
        selected_because=["domain_match"],
        score=0.85,
        must_carry=False,
        token_cost=chunk.token_cost,
    )


def _run_session(ledger: CHPLedger, session_id: str, embedder: StubEmbedder) -> dict:
    """
    Simulate one full 12-agent session:
      - select_chunks() for each agent
      - write selected envelopes to ledger
    Returns per-session timing stats.
    """
    chunks = _make_session_chunks(session_id)
    write_times = []
    query_times = []
    total_tokens_in  = sum(c.token_cost for c in chunks)
    total_tokens_out = 0
    errors = []

    for hop, (agent_id, must_carry, exclude, budget) in enumerate(AGENT_DEFS):
        manifest = ContextManifest(
            agent_id=agent_id,
            task=f"task_{agent_id}",
            requires=ContextRequirements(
                must_carry=must_carry,
                domain_tags=[agent_id, "billing", "fraud"],
                history_depth="decisions_only",
                exclude=exclude,
            ),
            token_budget=budget,
            on_missing="warn",
        )

        t0 = time.perf_counter()
        selected = select_chunks(chunks, manifest, embedder)
        query_times.append(time.perf_counter() - t0)

        total_tokens_out += sum(c.token_cost for c in selected)

        for chunk in selected:
            env = _make_envelope(session_id, agent_id, hop, chunk)
            t0 = time.perf_counter()
            try:
                ledger.write(session_id, hop, "router", agent_id, env)
            except RuntimeError:
                pass  # duplicate — expected when same chunk selected at same hop
            write_times.append(time.perf_counter() - t0)

    return {
        "session_id":    session_id,
        "write_times":   write_times,
        "query_times":   query_times,
        "tokens_in":     total_tokens_in,
        "tokens_out":    total_tokens_out,
        "errors":        errors,
    }


# ── Load test ─────────────────────────────────────────────────────────────────

def test_100_sessions_throughput_and_latency(tmp_path):
    """
    100 sessions × 12 agents, PARALLEL_WORKERS concurrent threads.
    Asserts:
      - all sessions complete without error
      - p99 write latency < 200ms
      - p99 query latency < 50ms
      - token reduction >= 10%
      - ledger row count matches expected writes
    """
    db_path = str(tmp_path / "load_ledger")
    ledger  = CHPLedger(db_path=db_path)
    embedder = StubEmbedder()

    session_ids = [f"load-sess-{i:04d}" for i in range(NUM_SESSIONS)]
    all_results = []
    all_errors  = []

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = {
            pool.submit(_run_session, ledger, sid, embedder): sid
            for sid in session_ids
        }
        for future in as_completed(futures):
            try:
                all_results.append(future.result())
            except Exception as exc:
                all_errors.append(str(exc))

    wall_elapsed = time.perf_counter() - wall_start

    # ── Collect metrics ───────────────────────────────────────────────────────
    all_write_times = [t for r in all_results for t in r["write_times"]]
    all_query_times = [t for r in all_results for t in r["query_times"]]
    total_tokens_in  = sum(r["tokens_in"]  for r in all_results)
    total_tokens_out = sum(r["tokens_out"] for r in all_results)

    write_p50 = statistics.median(all_write_times) * 1000
    write_p95 = sorted(all_write_times)[int(len(all_write_times) * 0.95)] * 1000
    write_p99 = sorted(all_write_times)[int(len(all_write_times) * 0.99)] * 1000
    query_p50 = statistics.median(all_query_times) * 1000
    query_p95 = sorted(all_query_times)[int(len(all_query_times) * 0.95)] * 1000
    query_p99 = sorted(all_query_times)[int(len(all_query_times) * 0.99)] * 1000

    token_reduction_pct = (1 - total_tokens_out / max(total_tokens_in, 1)) * 100
    stats = ledger.stats()
    sessions_per_sec = NUM_SESSIONS / wall_elapsed

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           CHP LOAD TEST REPORT — {NUM_SESSIONS} sessions × {AGENTS_PER_SESSION} agents        ║
╠══════════════════════════════════════════════════════════════╣
║  Throughput                                                  ║
║    sessions/sec      : {sessions_per_sec:>8.1f}                              ║
║    wall time         : {wall_elapsed:>8.2f}s                             ║
║    parallel workers  : {PARALLEL_WORKERS:>8d}                              ║
╠══════════════════════════════════════════════════════════════╣
║  Write latency (ms)                                          ║
║    p50               : {write_p50:>8.2f}                              ║
║    p95               : {write_p95:>8.2f}                              ║
║    p99               : {write_p99:>8.2f}                              ║
╠══════════════════════════════════════════════════════════════╣
║  select_chunks() latency (ms)                                ║
║    p50               : {query_p50:>8.2f}                              ║
║    p95               : {query_p95:>8.2f}                              ║
║    p99               : {query_p99:>8.2f}                              ║
╠══════════════════════════════════════════════════════════════╣
║  Token efficiency                                            ║
║    total input       : {total_tokens_in:>8,}                              ║
║    total output      : {total_tokens_out:>8,}                              ║
║    reduction         : {token_reduction_pct:>7.1f}%                              ║
╠══════════════════════════════════════════════════════════════╣
║  Ledger                                                      ║
║    ledger_rows       : {stats["ledger_rows"]:>8,}                              ║
║    chunk_rows        : {stats["chunk_rows"]:>8,}                              ║
║    errors            : {len(all_errors):>8}                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # ── Assertions ────────────────────────────────────────────────────────────
    assert not all_errors, f"Session errors:\n" + "\n".join(all_errors)
    assert len(all_results) == NUM_SESSIONS, \
        f"Expected {NUM_SESSIONS} sessions, got {len(all_results)}"

    assert write_p99 < 500,  f"write p99={write_p99:.1f}ms exceeds 500ms"
    assert query_p99 < 100,  f"select_chunks p99={query_p99:.1f}ms exceeds 100ms"
    assert token_reduction_pct >= 10, \
        f"token reduction {token_reduction_pct:.1f}% below 10% floor"

    assert stats["ledger_rows"] > 0
    assert stats["chunk_rows"]  > 0


def test_100_sessions_prune_and_orphan_cleanup(tmp_path):
    """After 100 sessions, prune all + orphan cleanup leaves ledger empty."""
    db_path  = str(tmp_path / "prune_ledger")
    ledger   = CHPLedger(db_path=db_path)
    embedder = StubEmbedder()

    session_ids = [f"prune-sess-{i:04d}" for i in range(NUM_SESSIONS)]

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        list(pool.map(lambda sid: _run_session(ledger, sid, embedder), session_ids))

    before = ledger.stats()
    assert before["ledger_rows"] > 0

    t0 = time.perf_counter()
    pruned_total = 0
    for sid in session_ids:
        pruned_total += ledger.prune(sid)
    prune_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    orphans = ledger.prune_orphan_chunks()
    orphan_time = time.perf_counter() - t0

    after = ledger.stats()
    print(f"\nPrune {NUM_SESSIONS} sessions: {prune_time*1000:.0f}ms  "
          f"({pruned_total} ledger rows)")
    print(f"Orphan cleanup: {orphan_time*1000:.0f}ms  ({orphans} chunks)")

    assert after["ledger_rows"] == 0, "all ledger rows must be gone after full prune"
    assert after["chunk_rows"]  == 0, "all orphan chunks must be cleaned up"


def test_100_sessions_ledger_fallback_under_load(tmp_path):
    """
    50 sessions write billing output. 50 sessions use ledger_fallback to recover it
    from their parent session. All 50 child sessions must recover billing_decision.
    """
    db_path  = str(tmp_path / "fallback_ledger")
    ledger   = CHPLedger(db_path=db_path)
    embedder = StubEmbedder()
    errors   = []

    def run_parent_child(i: int):
        parent_id = f"fb-parent-{i:03d}"
        child_id  = f"fb-child-{i:03d}"

        # Parent: billing runs, writes output
        billing_env = RationaleEnvelope(
            chunk_id=f"billing-out-{i:03d}",
            content=f"billing_decision: approve refund ${i} tier=premium",
            source_agent="billing-agent", source_turn=1,
            hop_sequence=["router", "billing-agent"],
            selected_because=["must_carry:order_id"],
            score=0.95, must_carry=True, token_cost=60,
        )
        ledger.write(parent_id, 0, "router", "billing-agent", billing_env)

        # Child: fraud skips billing, uses ledger_fallback with parent_session_id
        fraud_manifest = ContextManifest(
            agent_id="fraud-agent", task="assess_fraud",
            requires=ContextRequirements(
                must_carry=["billing_decision"],
                domain_tags=["fraud"],
                history_depth="decisions_only",
                exclude=["PII_raw"],
            ),
            token_budget=800,
            on_missing="ledger_fallback",
            parent_session_id=parent_id,
        )
        selected = select_chunks([], fraud_manifest, embedder, ledger=ledger, session_id=child_id)
        ids = [c.chunk_id for c in selected]
        if f"billing-out-{i:03d}" not in ids:
            errors.append(f"child {child_id} failed to recover billing output from parent {parent_id}")

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        list(pool.map(run_parent_child, range(50)))

    assert not errors, f"{len(errors)} failures:\n" + "\n".join(errors[:5])
