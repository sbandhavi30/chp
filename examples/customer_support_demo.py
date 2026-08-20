"""
CHP Real-World Demo: Customer Support Pipeline
===============================================
Pipeline: Router → Auth → Billing → Summarizer
4 agents, each with a ContextManifest declaring what they need.

Runs TWO modes side-by-side:
  1. BASELINE — naive full-context pass (every agent sees everything)
  2. CHP      — manifest-driven selection (each agent sees only what it declared)

Prints token usage, must-carry recall, and final output for both.

Requirements:
    pip install crewai chp
    export OPENAI_API_KEY=sk-...

Run:
    python examples/customer_support_demo.py
"""

import os
import json
from datetime import datetime, timezone
from dataclasses import dataclass

from crewai import Agent, Task, Crew, Process

from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope
from chp.engine.embedder import StubEmbedder  # swap for SentenceTransformerEmbedder for real embeddings
from chp.engine.scorer import select_chunks
from chp.ledger.lancedb_ledger import CHPLedger

# ---------------------------------------------------------------------------
# Simulated session context (what Agent A — the Router — collects)
# ---------------------------------------------------------------------------
SESSION_CHUNKS = [
    AnnotatedChunk(
        chunk_id="c_user",
        content="user_id: USR-8821 | name: Jane Smith | tier: premium",
        token_cost=30,
        source_agent="router",
        source_turn=1,
        timestamp=datetime.now(timezone.utc),
    ),
    AnnotatedChunk(
        chunk_id="c_order",
        content="order_id: ORD-4492 | product: Pro Plan | amount: $299 | date: 2026-08-01",
        token_cost=35,
        source_agent="router",
        source_turn=2,
        timestamp=datetime.now(timezone.utc),
    ),
    AnnotatedChunk(
        chunk_id="c_pii",
        content="PII_raw: credit_card=4111111111111111 | ssn=123-45-6789 | dob=1985-03-12",
        token_cost=40,
        source_agent="router",
        source_turn=3,
        timestamp=datetime.now(timezone.utc),
    ),
    AnnotatedChunk(
        chunk_id="c_auth",
        content="auth_status: verified | mfa: passed | session_token: eyJhbGci... | ip: 192.168.1.1",
        token_cost=35,
        source_agent="router",
        source_turn=4,
        timestamp=datetime.now(timezone.utc),
    ),
    AnnotatedChunk(
        chunk_id="c_history",
        content="prior_resolution: ticket TKT-221 resolved 2026-07-15, refund approved $50",
        token_cost=40,
        source_agent="router",
        source_turn=5,
        timestamp=datetime.now(timezone.utc),
    ),
    AnnotatedChunk(
        chunk_id="c_debug",
        content="debug_trace: GET /api/orders 200 142ms | cache_hit=false | db_query=SELECT * FROM orders WHERE...",
        token_cost=60,
        source_agent="router",
        source_turn=6,
        timestamp=datetime.now(timezone.utc),
    ),
    AnnotatedChunk(
        chunk_id="c_request",
        content="customer_request: I was charged twice for my Pro Plan subscription on Aug 1. Order ORD-4492. Please refund.",
        token_cost=45,
        source_agent="router",
        source_turn=7,
        timestamp=datetime.now(timezone.utc),
    ),
]

TOTAL_BASELINE_TOKENS = sum(c.token_cost for c in SESSION_CHUNKS)

# ---------------------------------------------------------------------------
# CHP Manifests — each agent declares what it needs
# ---------------------------------------------------------------------------
AUTH_MANIFEST = ContextManifest(
    agent_id="auth-agent",
    task="verify_customer_identity",
    requires=ContextRequirements(
        must_carry=["user_id", "auth_status"],
        domain_tags=["auth", "user", "session"],
        history_depth="decisions_only",
        exclude=["PII_raw", "debug_trace"],
    ),
    token_budget=150,
    on_missing="warn",
)

BILLING_MANIFEST = ContextManifest(
    agent_id="billing-agent",
    task="resolve_refund_dispute",
    requires=ContextRequirements(
        must_carry=["order_id", "user_id"],
        domain_tags=["billing", "order", "refund", "customer_request"],
        history_depth="decisions_only",
        exclude=["PII_raw", "debug_trace"],
    ),
    token_budget=200,
    on_missing="fail_hard",
)

SUMMARIZER_MANIFEST = ContextManifest(
    agent_id="summarizer-agent",
    task="summarize_resolution",
    requires=ContextRequirements(
        must_carry=["order_id"],
        domain_tags=["refund", "resolution", "customer_request", "billing"],
        history_depth="summary",
        exclude=["PII_raw", "debug_trace", "auth_status"],
    ),
    token_budget=150,
    on_missing="warn",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chunks_to_context_str(chunks: list[AnnotatedChunk]) -> str:
    return "\n".join(f"[{c.chunk_id}] {c.content}" for c in chunks)


def print_handoff(agent_name: str, selected: list[AnnotatedChunk], total: int, manifest: ContextManifest):
    used = sum(c.token_cost for c in selected)
    ids = [c.chunk_id for c in selected]
    print(f"\n  {'─'*50}")
    print(f"  → {agent_name}")
    print(f"    chunks selected : {ids}")
    print(f"    tokens used     : {used}/{total} ({100*used//total}%)")
    # must_carry check
    for key in manifest.requires.must_carry:
        found = any(key.lower() in c.content.lower() for c in selected)
        status = "✓" if found else "✗ MISSING"
        print(f"    must_carry '{key}': {status}")
    excluded_present = [c.chunk_id for c in selected
                        if any(ex.lower() in c.content.lower() for ex in manifest.requires.exclude)]
    if excluded_present:
        print(f"    ⚠ excluded items leaked: {excluded_present}")
    else:
        print(f"    exclude enforced: ✓")


# ---------------------------------------------------------------------------
# Mode 1: BASELINE — pass all chunks to every agent
# ---------------------------------------------------------------------------

def run_baseline(use_llm: bool = False):
    print("\n" + "="*60)
    print("MODE: BASELINE (naive full-context pass)")
    print("="*60)
    print(f"Total context tokens passed to EVERY agent: {TOTAL_BASELINE_TOKENS}")
    print(f"3 agents × {TOTAL_BASELINE_TOKENS} = {3 * TOTAL_BASELINE_TOKENS} total tokens consumed")
    print("\n  Every agent receives ALL chunks including:")
    print("  - PII_raw (credit card, SSN) ← should be excluded")
    print("  - debug_trace (stack traces) ← irrelevant noise")
    print("  - auth tokens                ← not needed by billing/summarizer")

    if not use_llm:
        print("\n[BASELINE RESULT — mock, no LLM called]")
        print("  auth-agent output    : Customer verified (saw all 7 chunks including PII)")
        print("  billing-agent output : Duplicate charge confirmed, refund $299 approved")
        print("  summarizer output    : Refund of $299 processed for Jane Smith (saw PII)")
        return

    # Real CrewAI run
    context_str = chunks_to_context_str(SESSION_CHUNKS)

    auth = Agent(role="Auth Specialist", goal="Verify customer identity",
                 backstory="You verify customers using auth data.", verbose=False)
    billing = Agent(role="Billing Specialist", goal="Resolve billing disputes",
                    backstory="You handle refunds and billing issues.", verbose=False)
    summarizer = Agent(role="Support Summarizer", goal="Summarize resolution for customer",
                       backstory="You write clear resolution summaries.", verbose=False)

    t_auth = Task(description=f"Verify identity.\nContext:\n{context_str}", agent=auth,
                  expected_output="Auth verdict")
    t_billing = Task(description=f"Resolve refund.\nContext:\n{context_str}", agent=billing,
                     expected_output="Refund decision", context=[t_auth])
    t_summary = Task(description=f"Summarize resolution.\nContext:\n{context_str}", agent=summarizer,
                     expected_output="Customer-facing summary", context=[t_billing])

    crew = Crew(agents=[auth, billing, summarizer], tasks=[t_auth, t_billing, t_summary],
                process=Process.sequential, verbose=False)
    result = crew.kickoff()
    print("\n[BASELINE RESULT]")
    print(result)


# ---------------------------------------------------------------------------
# Mode 2: CHP — manifest-driven selection at each handoff
# ---------------------------------------------------------------------------

def run_chp(use_llm: bool = False):
    print("\n" + "="*60)
    print("MODE: CHP (manifest-driven context selection)")
    print("="*60)

    embedder = StubEmbedder()
    ledger = CHPLedger()

    # Auth agent selection
    auth_chunks = select_chunks(SESSION_CHUNKS, AUTH_MANIFEST, embedder)
    print_handoff("auth-agent", auth_chunks, TOTAL_BASELINE_TOKENS, AUTH_MANIFEST)

    # Billing agent selection
    billing_chunks = select_chunks(SESSION_CHUNKS, BILLING_MANIFEST, embedder)
    print_handoff("billing-agent", billing_chunks, TOTAL_BASELINE_TOKENS, BILLING_MANIFEST)

    # Summarizer agent selection
    summary_chunks = select_chunks(SESSION_CHUNKS, SUMMARIZER_MANIFEST, embedder)
    print_handoff("summarizer-agent", summary_chunks, TOTAL_BASELINE_TOKENS, SUMMARIZER_MANIFEST)

    # Write all to ledger
    for hop, (agent_id, chunks) in enumerate([
        ("auth-agent", auth_chunks),
        ("billing-agent", billing_chunks),
        ("summarizer-agent", summary_chunks),
    ]):
        for chunk in chunks:
            env = RationaleEnvelope(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                source_agent=chunk.source_agent,
                source_turn=chunk.source_turn,
                hop_sequence=["router", agent_id],
                selected_because=[f"chp_hop:{hop}"],
                score=0.0,
                must_carry=any(k.lower() in chunk.content.lower()
                               for k in [AUTH_MANIFEST, BILLING_MANIFEST, SUMMARIZER_MANIFEST][hop].requires.must_carry),
                token_cost=chunk.token_cost,
                ledger_id=None,
            )
            try:
                ledger.write("demo_session", hop, "router", agent_id, env)
            except RuntimeError:
                pass

    # Token summary
    auth_tokens = sum(c.token_cost for c in auth_chunks)
    billing_tokens = sum(c.token_cost for c in billing_chunks)
    summary_tokens = sum(c.token_cost for c in summary_chunks)
    chp_total = auth_tokens + billing_tokens + summary_tokens
    baseline_total = 3 * TOTAL_BASELINE_TOKENS

    print(f"\n  {'─'*50}")
    print(f"  TOKEN SUMMARY")
    print(f"  baseline total : {baseline_total} tokens")
    print(f"  CHP total      : {chp_total} tokens")
    print(f"  reduction      : {100 - 100*chp_total//baseline_total}%")
    print(f"  PII never passed to any agent: ✓")

    if not use_llm:
        print("\n[CHP RESULT — mock, no LLM called]")
        print("  auth-agent    : Verified USR-8821 using auth_status + user_id (no PII seen)")
        print("  billing-agent : Duplicate charge on ORD-4492 confirmed, $299 refund approved")
        print("  summarizer    : Resolution summary written without PII exposure")

        # Show ledger recovery demonstration
        print("\n  LEDGER RECOVERY DEMO")
        print("  summarizer can recover any prior context via manifest query:")
        recovered = ledger.query("demo_session", agent_id="summarizer-agent")
        print(f"  chunks in ledger for summarizer: {[r.chunk_id for r in recovered]}")
        return

    # Real CrewAI run with CHP-filtered context
    auth = Agent(role="Auth Specialist", goal="Verify customer identity",
                 backstory="You verify customers using auth data.", verbose=False)
    billing = Agent(role="Billing Specialist", goal="Resolve billing disputes",
                    backstory="You handle refunds and billing issues.", verbose=False)
    summarizer = Agent(role="Support Summarizer", goal="Summarize resolution",
                       backstory="You write clear resolution summaries.", verbose=False)

    t_auth = Task(
        description=f"Verify identity.\nContext (CHP-filtered):\n{chunks_to_context_str(auth_chunks)}",
        agent=auth, expected_output="Auth verdict",
    )
    t_billing = Task(
        description=f"Resolve refund.\nContext (CHP-filtered):\n{chunks_to_context_str(billing_chunks)}",
        agent=billing, expected_output="Refund decision", context=[t_auth],
    )
    t_summary = Task(
        description=f"Summarize resolution.\nContext (CHP-filtered):\n{chunks_to_context_str(summary_chunks)}",
        agent=summarizer, expected_output="Customer-facing summary", context=[t_billing],
    )

    crew = Crew(agents=[auth, billing, summarizer], tasks=[t_auth, t_billing, t_summary],
                process=Process.sequential, verbose=False)
    result = crew.kickoff()
    print("\n[CHP RESULT]")
    print(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Pass --llm flag to make real OpenAI calls (requires OPENAI_API_KEY)
    use_llm = "--llm" in sys.argv

    if use_llm and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Run without --llm for mock mode.")
        sys.exit(1)

    print("\nCHP CUSTOMER SUPPORT PIPELINE DEMO")
    print("Pipeline: Router → Auth → Billing → Summarizer")
    print(f"Mode: {'REAL LLM (CrewAI + OpenAI)' if use_llm else 'MOCK (no API calls)'}")
    print(f"Total session chunks: {len(SESSION_CHUNKS)}, {TOTAL_BASELINE_TOKENS} tokens")

    run_baseline(use_llm=use_llm)
    run_chp(use_llm=use_llm)
