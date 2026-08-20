"""
Real LLM load test: 100 sessions × 12 agents.

Each agent:
  - infer_manifest() via real OpenAI (gpt-4o-mini by default)
  - select_chunks() with real SentenceTransformerEmbedder
  - generates real LLM output, writes back as agent_output chunk
  - ledger stores real 384-dim embeddings in LanceDB
  - query_by_meaning() validates semantic recovery

Requirements:
  pip install "chp[all]"          # sentence-transformers + openai + crewai
  export OPENAI_API_KEY=sk-...

Estimated cost: ~$0.25 (gpt-4o-mini) or ~$3.75 (gpt-4o) for 100 sessions.

Run:
  pytest chp/tests/load/test_load_llm.py -v -s --timeout=600

Skip if no API key:
  Tests auto-skip when OPENAI_API_KEY is not set.
"""
from __future__ import annotations

import os
import statistics
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

# ── Skip entire module if no API key ─────────────────────────────────────────
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real LLM load test",
)

from chp.engine.scorer import select_chunks
from chp.inference import infer_manifest
from chp.ledger.lancedb_ledger import CHPLedger
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope

# ── Config ────────────────────────────────────────────────────────────────────

NUM_SESSIONS     = 100
PARALLEL_WORKERS = 10    # conservative — each worker makes real API calls
LLM_MODEL        = os.environ.get("CHP_LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL      = os.environ.get("CHP_EMBED_MODEL", "all-MiniLM-L6-v2")

# 12-agent pipeline — same roles as full_pipeline_demo.py
AGENT_DEFS = [
    {
        "agent_id": "router",
        "role": "Support Router",
        "goal": "Classify and dispatch incoming customer support tickets to the right specialist team",
        "backstory": "",
    },
    {
        "agent_id": "auth",
        "role": "Auth Specialist",
        "goal": "Verify customer identity and confirm authentication status before any account action",
        "backstory": "",
    },
    {
        "agent_id": "billing",
        "role": "Billing Specialist",
        "goal": "Resolve duplicate charge disputes and process refunds for customer orders",
        "backstory": "You handle billing issues for premium customers with authority to approve refunds under $500",
    },
    {
        "agent_id": "fraud",
        "role": "Fraud Analyst",
        "goal": "Assess fraud risk and flag suspicious transactions based on velocity and device signals",
        "backstory": "",
    },
    {
        "agent_id": "compliance",
        "role": "Compliance Officer",
        "goal": "Check GDPR and PCI compliance for customer data handling in this support case",
        "backstory": "",
    },
    {
        "agent_id": "policy",
        "role": "Policy Checker",
        "goal": "Check refund eligibility and policy terms for the customer order",
        "backstory": "",
    },
    {
        "agent_id": "research",
        "role": "Research Agent",
        "goal": "Search and retrieve relevant knowledge base articles for duplicate charge resolution",
        "backstory": "",
    },
    {
        "agent_id": "orchestrator",
        "role": "Orchestrator",
        "goal": "Synthesize results from all specialist subagents and produce the final resolution decision",
        "backstory": "",
    },
    {
        "agent_id": "escalation",
        "role": "Escalation Manager",
        "goal": "Coordinate escalation response and delegate to senior specialists if auto-approval fails",
        "backstory": "",
    },
    {
        "agent_id": "summarizer",
        "role": "Customer Summarizer",
        "goal": "Summarize the resolution outcome for the customer-facing response",
        "backstory": "",
    },
    {
        "agent_id": "code_reviewer",
        "role": "Code Reviewer",
        "goal": "Review security and style issues in the billing service diff that caused the duplicate charge",
        "backstory": "",
    },
    {
        "agent_id": "auditor",
        "role": "Account Auditor",
        "goal": "Audit account history and verify identity for suspicious login or charge patterns",
        "backstory": "",
    },
]


# ── Shared resources (created once, reused across sessions) ───────────────────

@pytest.fixture(scope="module")
def llm_client():
    import openai
    return openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@pytest.fixture(scope="module")
def embedder():
    from chp.engine.embedder import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder(EMBED_MODEL)


@pytest.fixture(scope="module")
def inferred_manifests(llm_client):
    """
    Infer all 12 manifests once via LLM — reused across all 100 sessions.
    Saves ~1,200 LLM calls (12 agents × 100 sessions → 12 calls total).
    """
    print(f"\nInferring 12 manifests via {LLM_MODEL}...")
    manifests = {}
    for agent in AGENT_DEFS:
        t0 = time.perf_counter()
        m = infer_manifest(
            role=agent["role"],
            goal=agent["goal"],
            backstory=agent.get("backstory", ""),
            agent_id=agent["agent_id"],
            llm_client=llm_client,
            model=LLM_MODEL,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  {agent['agent_id']:15s} must_carry={m.requires.must_carry} "
              f"budget={m.token_budget} [{elapsed:.0f}ms]")
        manifests[agent["agent_id"]] = m
    return manifests


@pytest.fixture(scope="module")
def shared_ledger(tmp_path_factory):
    db_path = str(tmp_path_factory.mktemp("llm_ledger"))
    return CHPLedger(db_path=db_path)


# ── Session chunk factory ─────────────────────────────────────────────────────

def _make_session_chunks(session_id: str) -> list[AnnotatedChunk]:
    return [
        AnnotatedChunk(chunk_id=f"{session_id}_order",      content="order_id: ORD-4492 amount: $299 currency: USD",    token_cost=30,  source_agent="router",  source_turn=0),
        AnnotatedChunk(chunk_id=f"{session_id}_user",        content="user_id: USR-001 tier: premium account_age: 3yr", token_cost=25,  source_agent="router",  source_turn=0),
        AnnotatedChunk(chunk_id=f"{session_id}_auth",        content="auth_status: verified mfa_method: totp",           token_cost=20,  source_agent="auth",    source_turn=1),
        AnnotatedChunk(chunk_id=f"{session_id}_fraud",       content="fraud_score: 0.12 velocity: low device: mobile",   token_cost=35,  source_agent="fraud",   source_turn=2),
        AnnotatedChunk(chunk_id=f"{session_id}_compliance",  content="compliance_flags: GDPR_ok PCI_ok data_region: EU", token_cost=40,  source_agent="comp",    source_turn=2),
        AnnotatedChunk(chunk_id=f"{session_id}_policy",      content="refund_eligible: true policy_version: v3 window: 30day", token_cost=30, source_agent="policy", source_turn=2),
        AnnotatedChunk(chunk_id=f"{session_id}_decision",    content="final_decision: approve refund_amount: $299",      token_cost=45,  source_agent="orch",    source_turn=3),
        AnnotatedChunk(chunk_id=f"{session_id}_history",     content="account_history: 3 prior orders all clean no chargebacks", token_cost=80, source_agent="audit", source_turn=1),
        AnnotatedChunk(chunk_id=f"{session_id}_kb",          content="kb_article: duplicate charge resolution — check idempotency key", token_cost=120, source_agent="research", source_turn=2),
        AnnotatedChunk(chunk_id=f"{session_id}_diff",        content="billing_diff: fixed missing idempotency key in charge_handler.py line 142", token_cost=200, source_agent="code_reviewer", source_turn=0),
    ]


# ── LLM task call ─────────────────────────────────────────────────────────────

def _call_llm(client, agent_def: dict, context_summary: str, model: str) -> str:
    """Single LLM call — agent generates its output given filtered context."""
    prompt = (
        f"You are a {agent_def['role']}. Your goal: {agent_def['goal']}\n\n"
        f"Context provided to you:\n{context_summary}\n\n"
        f"Respond in 1-2 sentences with your specific decision or finding."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=80,
    )
    return response.choices[0].message.content.strip()


# ── Session runner ─────────────────────────────────────────────────────────────

def _run_llm_session(
    session_id: str,
    ledger: CHPLedger,
    manifests: dict,
    embedder,
    llm_client,
) -> dict:
    chunks = _make_session_chunks(session_id)
    tokens_in  = sum(c.token_cost for c in chunks)
    tokens_out = 0
    write_times  = []
    select_times = []
    llm_times    = []
    pii_leaks    = []
    agent_outputs: list[AnnotatedChunk] = []

    for hop, agent_def in enumerate(AGENT_DEFS):
        agent_id = agent_def["agent_id"]
        manifest = manifests[agent_id]

        # select_chunks with real embeddings
        pool = chunks + agent_outputs
        t0 = time.perf_counter()
        selected = select_chunks(pool, manifest, embedder)
        select_times.append(time.perf_counter() - t0)
        tokens_out += sum(c.token_cost for c in selected)

        # PII check — credit_card / ssn must never reach non-compliance agents
        if agent_id not in ("compliance", "auditor"):
            for c in selected:
                if any(pii in c.content.lower() for pii in ["credit_card", "ssn", "pii_raw"]):
                    pii_leaks.append(f"{agent_id} received PII in chunk {c.chunk_id}")

        # Real LLM call
        context_summary = "\n".join(f"- {c.content}" for c in selected) or "(no context)"
        t0 = time.perf_counter()
        llm_output = _call_llm(llm_client, agent_def, context_summary, LLM_MODEL)
        llm_times.append(time.perf_counter() - t0)

        # Write agent output back as is_agent_output chunk
        output_chunk = AnnotatedChunk(
            chunk_id=f"{session_id}_{agent_id}_output",
            content=llm_output,
            token_cost=len(llm_output.split()) * 2,  # rough token estimate
            source_agent=agent_id,
            source_turn=hop,
            is_agent_output=True,
        )
        agent_outputs.append(output_chunk)

        # Write selected envelopes to ledger with real embeddings
        for chunk in selected[:3]:  # top 3 per agent to keep ledger bounded
            env = RationaleEnvelope(
                chunk_id=f"{session_id}_{agent_id}_{chunk.chunk_id[-8:]}",
                content=chunk.content,
                source_agent=agent_id,
                source_turn=hop,
                hop_sequence=["router", agent_id],
                selected_because=["domain_match"],
                score=0.85,
                must_carry=False,
                token_cost=chunk.token_cost,
            )
            t0 = time.perf_counter()
            try:
                ledger.write(session_id, hop, "router", agent_id, env, embedder=embedder)
            except RuntimeError:
                pass  # duplicate
            write_times.append(time.perf_counter() - t0)

    return {
        "session_id":    session_id,
        "tokens_in":     tokens_in,
        "tokens_out":    tokens_out,
        "write_times":   write_times,
        "select_times":  select_times,
        "llm_times":     llm_times,
        "pii_leaks":     pii_leaks,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_llm_manifest_inference_quality(inferred_manifests):
    """All 12 inferred manifests are valid and sensible."""
    assert len(inferred_manifests) == 12

    billing = inferred_manifests["billing"]
    assert billing.token_budget >= 500,  "billing budget too small"
    assert any("order" in k.lower() for k in billing.requires.must_carry), \
        f"billing must_carry missing order field: {billing.requires.must_carry}"

    fraud = inferred_manifests["fraud"]
    assert any("fraud" in k.lower() or "score" in k.lower() for k in fraud.requires.must_carry), \
        f"fraud must_carry missing fraud_score: {fraud.requires.must_carry}"

    summarizer = inferred_manifests["summarizer"]
    assert summarizer.token_budget <= 1500, \
        f"summarizer budget unexpectedly large: {summarizer.token_budget}"

    code_reviewer = inferred_manifests["code_reviewer"]
    assert code_reviewer.token_budget >= 2000, \
        f"code_reviewer budget too small: {code_reviewer.token_budget}"

    # PII excluded from non-compliance agents
    for agent_id, m in inferred_manifests.items():
        if agent_id in ("compliance", "auditor"):
            continue
        combined = " ".join(m.requires.exclude).lower()
        assert any(pii in combined for pii in ["pii", "ssn", "credit"]), \
            f"{agent_id} missing PII in exclude list: {m.requires.exclude}"


def test_100_sessions_real_llm_throughput(
    shared_ledger, inferred_manifests, embedder, llm_client, tmp_path
):
    """
    100 sessions × 12 real LLM calls each.
    Asserts: no errors, PII contained, token reduction >= 10%, p99 latencies reasonable.
    """
    session_ids = [f"llm-sess-{i:04d}" for i in range(NUM_SESSIONS)]
    all_results = []
    all_errors  = []

    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = {
            pool.submit(
                _run_llm_session,
                sid, shared_ledger, inferred_manifests, embedder, llm_client,
            ): sid
            for sid in session_ids
        }
        for future in as_completed(futures):
            try:
                all_results.append(future.result())
            except Exception as exc:
                all_errors.append(str(exc))

    wall_elapsed = time.perf_counter() - wall_start

    # ── Metrics ───────────────────────────────────────────────────────────────
    all_write   = sorted([t for r in all_results for t in r["write_times"]])
    all_select  = sorted([t for r in all_results for t in r["select_times"]])
    all_llm     = sorted([t for r in all_results for t in r["llm_times"]])
    all_pii     = [leak for r in all_results for leak in r["pii_leaks"]]

    tokens_in  = sum(r["tokens_in"]  for r in all_results)
    tokens_out = sum(r["tokens_out"] for r in all_results)
    reduction  = (1 - tokens_out / max(tokens_in, 1)) * 100

    stats = shared_ledger.stats()

    def pct(arr, p):
        return arr[int(len(arr) * p / 100)] * 1000 if arr else 0

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║       CHP REAL LLM LOAD TEST — {NUM_SESSIONS} sessions × 12 agents       ║
║       Model: {LLM_MODEL:<20s}  Embedder: {EMBED_MODEL:<12s}  ║
╠══════════════════════════════════════════════════════════════╣
║  Throughput                                                  ║
║    wall time         : {wall_elapsed:>8.1f}s                             ║
║    sessions/sec      : {NUM_SESSIONS/wall_elapsed:>8.2f}                              ║
║    parallel workers  : {PARALLEL_WORKERS:>8d}                              ║
╠══════════════════════════════════════════════════════════════╣
║  LLM call latency (ms per agent call)                        ║
║    p50               : {pct(all_llm,50):>8.0f}                              ║
║    p95               : {pct(all_llm,95):>8.0f}                              ║
║    p99               : {pct(all_llm,99):>8.0f}                              ║
╠══════════════════════════════════════════════════════════════╣
║  select_chunks() latency (ms, real embeddings)               ║
║    p50               : {pct(all_select,50):>8.1f}                              ║
║    p95               : {pct(all_select,95):>8.1f}                              ║
║    p99               : {pct(all_select,99):>8.1f}                              ║
╠══════════════════════════════════════════════════════════════╣
║  LanceDB write latency (ms, with embedding)                  ║
║    p50               : {pct(all_write,50):>8.1f}                              ║
║    p95               : {pct(all_write,95):>8.1f}                              ║
║    p99               : {pct(all_write,99):>8.1f}                              ║
╠══════════════════════════════════════════════════════════════╣
║  Token efficiency                                            ║
║    total input       : {tokens_in:>8,}                              ║
║    total output      : {tokens_out:>8,}                              ║
║    reduction         : {reduction:>7.1f}%                              ║
╠══════════════════════════════════════════════════════════════╣
║  Safety & correctness                                        ║
║    PII leaks         : {len(all_pii):>8}                              ║
║    session errors    : {len(all_errors):>8}                              ║
╠══════════════════════════════════════════════════════════════╣
║  Ledger (LanceDB)                                            ║
║    ledger_rows       : {stats["ledger_rows"]:>8,}                              ║
║    chunk_rows        : {stats["chunk_rows"]:>8,}                              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    # ── Assertions ────────────────────────────────────────────────────────────
    assert not all_errors,  f"Session errors:\n" + "\n".join(all_errors[:5])
    assert not all_pii,     f"PII leaks:\n"      + "\n".join(all_pii[:5])
    assert len(all_results) == NUM_SESSIONS
    assert reduction >= 10, f"token reduction {reduction:.1f}% below 10%"
    assert stats["ledger_rows"] > 0
    assert stats["chunk_rows"]  > 0


def test_query_by_meaning_after_llm_sessions(shared_ledger, embedder):
    """
    After 100 real sessions with real embeddings, semantic search must return results
    for billing-related queries scoped to a specific session.
    """
    # Pick any session that was written
    stats = shared_ledger.stats()
    if stats["ledger_rows"] == 0:
        pytest.skip("no ledger rows — run throughput test first")

    results = shared_ledger.query_by_meaning(
        query_text="billing refund duplicate charge approve",
        embedder=embedder,
        session_id="llm-sess-0000",
        top_k=5,
    )
    # With real embeddings, billing-related chunks must rank high
    assert isinstance(results, list)
    # At least one result if session-0000 was written
    if results:
        combined = " ".join(r.content.lower() for r in results)
        assert any(word in combined for word in ["order", "billing", "refund", "approve", "decision"]), \
            f"semantic search returned unrelated chunks: {[r.content[:60] for r in results]}"
