"""
CHP Full Pipeline Demo — 12 Agents, 3 Topologies, Production Ledger
=====================================================================

Simulates a complete customer support resolution pipeline:

  Phase 1 — Sequential triage (Hop 0-2)
    Router → Auth Specialist → Support Router

  Phase 2 — Fan-out: 5 parallel specialists (Hop 3)
    Orchestrator dispatches simultaneously to:
      → Billing Specialist
      → Fraud Analyst
      → Compliance Officer
      → Policy Checker
      → Research Agent

  Phase 3 — Fan-in: Orchestrator synthesizes (Hop 4)
    Orchestrator collects all 5 results from ledger

  Phase 4 — Sequential resolution (Hop 5-6)
    Escalation Manager → Customer Summarizer

  Throughout:
    • Each agent auto-declares its manifest via infer_manifest()
    • Production ledger: content deduplication, TTL prune, semantic query
    • Baseline vs CHP token comparison printed at every hop

Run:
    python examples/full_pipeline_demo.py
    python examples/full_pipeline_demo.py --embeddings   # real semantic vectors (needs sentence-transformers)
    python examples/full_pipeline_demo.py --no-prune     # keep ledger alive after demo
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from chp.engine.embedder import StubEmbedder
from chp.engine.scorer import select_chunks
from chp.inference import infer_manifest
from chp.ledger.lancedb_ledger import CHPLedger
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="CHP Full Pipeline Demo")
parser.add_argument("--embeddings", action="store_true",
                    help="Use sentence-transformers for real semantic vectors")
parser.add_argument("--no-prune", action="store_true",
                    help="Skip ledger prune at the end (keep data for inspection)")
parser.add_argument("--llm", action="store_true",
                    help="Run real CrewAI + OpenAI agents (requires OPENAI_API_KEY)")
parser.add_argument("--model", default="gpt-4o-mini",
                    help="OpenAI model to use with --llm (default: gpt-4o-mini)")
ARGS = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# LLM validation
# ─────────────────────────────────────────────────────────────────────────────

if ARGS.llm:
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set.")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    try:
        from crewai import Agent, Task, Crew, Process as CrewProcess
    except ImportError:
        print("ERROR: crewai not installed.  pip install crewai")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Embedder
# ─────────────────────────────────────────────────────────────────────────────

if ARGS.embeddings:
    try:
        from chp.engine.embedder import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()
        print("  Embedder: sentence-transformers (all-MiniLM-L6-v2)")
    except ImportError:
        print("  WARNING: sentence-transformers not installed. pip install 'chp[embeddings]'")
        print("  Falling back to StubEmbedder.\n")
        embedder = StubEmbedder()
else:
    embedder = StubEmbedder()
    print("  Embedder: StubEmbedder (zero vectors — semantic ranking disabled)")
    print("  Use --embeddings for real semantic selection.\n")

# ─────────────────────────────────────────────────────────────────────────────
# Session context — 14 chunks, 510 tokens
# This is what the router collected at session start.
# ─────────────────────────────────────────────────────────────────────────────

SESSION_ID = f"demo-{int(time.time())}"

SESSION = [
    AnnotatedChunk(chunk_id="c_user",       content="user_id: USR-8821 | tier: premium | account_age: 3yr",                   token_cost=30,  source_agent="router", source_turn=1,  timestamp=datetime.now(timezone.utc)),
    AnnotatedChunk(chunk_id="c_order",      content="order_id: ORD-4492 | amount: $299 | date: 2026-08-01 | plan: Pro",        token_cost=35,  source_agent="router", source_turn=2,  timestamp=datetime.now(timezone.utc)),
    AnnotatedChunk(chunk_id="c_pii",        content="PII_raw: credit_card=4111111111111111 | ssn=123-45-6789",                 token_cost=40,  source_agent="router", source_turn=3,  timestamp=datetime.now(timezone.utc)),
    AnnotatedChunk(chunk_id="c_auth",       content="auth_status: verified | mfa: passed | session_token: eyJhbGciOi...",      token_cost=35,  source_agent="router", source_turn=4,  timestamp=datetime.now(timezone.utc)),
    AnnotatedChunk(chunk_id="c_fraud",      content="fraud_score: 0.12 | ip_risk: low | device: known | velocity: normal",     token_cost=40,  source_agent="router", source_turn=5,  timestamp=datetime.now(timezone.utc)),
    AnnotatedChunk(chunk_id="c_compliance", content="compliance_flags: GDPR=ok | PCI=flagged_raw_card | SOX=ok | residency=EU",token_cost=40,  source_agent="router", source_turn=6,  timestamp=datetime.now(timezone.utc)),
    AnnotatedChunk(chunk_id="c_policy",     content="refund_policy: Pro Plan 30-day full refund | duplicate_charge: auto-approve <$500", token_cost=45, source_agent="router", source_turn=7, timestamp=datetime.now(timezone.utc)),
    AnnotatedChunk(chunk_id="c_request",    content="customer_request: charged twice for Pro Plan Aug 1, order ORD-4492, refund $299", token_cost=45, source_agent="router", source_turn=8, timestamp=datetime.now(timezone.utc)),
    AnnotatedChunk(chunk_id="c_debug",      content="debug_trace: POST /api/charge 200 | idempotency_key=missing | charge_id=ch_001 ch_002", token_cost=55, source_agent="router", source_turn=9, timestamp=datetime.now(timezone.utc)),
    AnnotatedChunk(chunk_id="c_history",    content="prior_tickets: TKT-221 refund $50 approved 2026-07-15 | TKT-198 resolved 2026-05",  token_cost=45, source_agent="router", source_turn=10, timestamp=datetime.now(timezone.utc)),
    AnnotatedChunk(chunk_id="c_diff",       content="git diff: billing_service.py idempotency fix +charge_v2 -charge_v1",      token_cost=50,  source_agent="router", source_turn=11, timestamp=datetime.now(timezone.utc)),
    AnnotatedChunk(chunk_id="c_kb",         content="knowledge_base: duplicate charge resolution SOP — verify idempotency key, check charge_ids, auto-approve if <$500", token_cost=55, source_agent="router", source_turn=12, timestamp=datetime.now(timezone.utc)),
    AnnotatedChunk(chunk_id="c_escalation", content="escalation_rules: premium tier SLA 2hr | duplicate_charge: auto-approve, no manager needed if <$500", token_cost=40, source_agent="router", source_turn=13, timestamp=datetime.now(timezone.utc)),
    AnnotatedChunk(chunk_id="c_outcome_tpl",content="outcome_template: Dear {name}, your refund of {amount} has been {status}.",  token_cost=35, source_agent="router", source_turn=14, timestamp=datetime.now(timezone.utc)),
]

TOTAL_TOKENS = sum(c.token_cost for c in SESSION)

# ─────────────────────────────────────────────────────────────────────────────
# 12 Agent definitions — role + goal, manifests auto-inferred
# ─────────────────────────────────────────────────────────────────────────────

AGENT_DEFS = {
    # Phase 1 — sequential triage
    "router": {
        "role": "Support Router",
        "goal": "Classify and dispatch incoming customer support tickets to the right specialist team",
    },
    "auth": {
        "role": "Auth Specialist",
        "goal": "Verify customer identity and confirm authentication status before any account action",
    },
    # Phase 2 — fan-out specialists
    "billing": {
        "role": "Billing Specialist",
        "goal": "Resolve duplicate charge disputes and process refunds for customer orders",
        "backstory": "You handle billing issues for premium customers with authority to approve refunds under $500",
    },
    "fraud": {
        "role": "Fraud Analyst",
        "goal": "Assess fraud risk and flag suspicious transactions based on velocity and device signals",
    },
    "compliance": {
        "role": "Compliance Officer",
        "goal": "Check GDPR and PCI compliance for customer data handling in this support case",
    },
    "policy": {
        "role": "Policy Checker",
        "goal": "Check refund eligibility and policy terms for the customer order",
    },
    "research": {
        "role": "Research Agent",
        "goal": "Search and retrieve relevant knowledge base articles for duplicate charge resolution",
    },
    # Phase 3 — orchestrator (fan-in)
    "orchestrator": {
        "role": "Orchestrator",
        "goal": "Synthesize results from all specialist subagents and produce the final resolution decision",
    },
    # Phase 4 — sequential resolution
    "escalation": {
        "role": "Escalation Manager",
        "goal": "Coordinate escalation response and delegate to senior specialists if auto-approval fails",
    },
    "summarizer": {
        "role": "Customer Summarizer",
        "goal": "Summarize the resolution outcome for the customer-facing response",
    },
    # Background agents (run async throughout)
    "code_reviewer": {
        "role": "Code Reviewer",
        "goal": "Review security and style issues in the billing service diff that caused the duplicate charge",
    },
    "auditor": {
        "role": "Account Auditor",
        "goal": "Audit account history and verify identity for suspicious login or charge patterns",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_envelope(chunk: AnnotatedChunk, from_agent: str, to_agent: str, hop: int) -> RationaleEnvelope:
    return RationaleEnvelope(
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        source_agent=chunk.source_agent,
        source_turn=chunk.source_turn,
        hop_sequence=[from_agent, to_agent],
        selected_because=[f"chp_selected_hop:{hop}"],
        score=0.7,
        must_carry=False,
        token_cost=chunk.token_cost,
    )


def run_agent(
    agent_key: str,
    chunks: list[AnnotatedChunk],
    hop: int,
    from_agent: str,
    ledger: CHPLedger,
    all_chunks: list[AnnotatedChunk],
) -> tuple[str, list[AnnotatedChunk], int]:
    """
    Run one agent:
      1. Auto-infer manifest from role/goal
      2. Score + select context
      3. Write to production ledger (with dedup)
      4. Return selected chunks as this agent's output
    """
    defn = AGENT_DEFS[agent_key]
    manifest = infer_manifest(**defn)
    selected = select_chunks(all_chunks, manifest, embedder)

    for chunk in selected:
        env = _make_envelope(chunk, from_agent, manifest.agent_id, hop)
        try:
            ledger.write(SESSION_ID, hop, from_agent, manifest.agent_id, env, embedder=embedder)
        except RuntimeError:
            pass  # same chunk already written to this hop by another path

    tokens = sum(c.token_cost for c in selected)
    return agent_key, selected, tokens


def _chunks_to_str(chunks: list[AnnotatedChunk]) -> str:
    return "\n".join(f"[{c.chunk_id}] {c.content}" for c in chunks)


def _run_crewai_agent(
    role: str,
    goal: str,
    backstory: str,
    context_chunks: list[AnnotatedChunk],
    task_description: str,
    expected_output: str,
    prior_task_result: str = "",
) -> str:
    """Run a single CrewAI agent with CHP-filtered context. Returns agent output string."""
    ctx = _chunks_to_str(context_chunks)
    full_desc = task_description
    if prior_task_result:
        full_desc += f"\n\nPrior agent output:\n{prior_task_result}"
    full_desc += f"\n\nContext (CHP-filtered — only what this agent declared it needs):\n{ctx}"

    agent = Agent(role=role, goal=goal, backstory=backstory, verbose=False,
                  llm=f"openai/{ARGS.model}")
    task  = Task(description=full_desc, agent=agent, expected_output=expected_output)
    crew  = Crew(agents=[agent], tasks=[task], process=CrewProcess.sequential, verbose=False)
    result = crew.kickoff()
    return str(result)


def _separator(char="─", width=72):
    print(char * width)


def _print_agent_result(agent_key: str, selected: list[AnnotatedChunk], budget: int, hop: int):
    defn   = AGENT_DEFS[agent_key]
    tokens = sum(c.token_cost for c in selected)
    ids    = [c.chunk_id for c in selected]
    pct    = (TOTAL_TOKENS - tokens) / TOTAL_TOKENS * 100
    pii_ok = "c_pii" not in ids
    dbg_ok = "c_debug" not in ids
    print(
        f"  [{hop}] {defn['role']:<28} "
        f"tokens={tokens:4d}/{TOTAL_TOKENS}  "
        f"saved={pct:4.0f}%  "
        f"PII={'✓' if pii_ok else '✗'}  DBG={'✓' if dbg_ok else '✗'}  "
        f"chunks={ids}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ledger = CHPLedger()
    print()
    _separator("═")
    print("  CHP Full Pipeline Demo — 12 Agents, 3 Topologies, Production Ledger")
    _separator("═")
    print(f"  Session : {SESSION_ID}")
    print(f"  Chunks  : {len(SESSION)} | Total tokens: {TOTAL_TOKENS}")
    print(f"  Agents  : {len(AGENT_DEFS)}")
    print(f"  Mode    : {'REAL LLM — CrewAI + ' + ARGS.model if ARGS.llm else 'MOCK — no LLM calls'}")
    print()

    total_chp_tokens   = 0
    total_baseline     = TOTAL_TOKENS * len(AGENT_DEFS)
    llm_outputs: dict[str, str] = {}   # agent_key → LLM response text

    # ── Phase 1: Sequential triage ─────────────────────────────────────────
    _separator()
    print("  PHASE 1 — Sequential Triage  (Router → Auth)")
    _separator()

    for hop, (agent_key, from_key) in enumerate([
        ("router", "session_start"),
        ("auth",   "router"),
    ]):
        _, selected, tokens = run_agent(agent_key, SESSION, hop, from_key, ledger, SESSION)
        _print_agent_result(agent_key, selected, infer_manifest(**AGENT_DEFS[agent_key]).token_budget, hop)
        total_chp_tokens += tokens

        if ARGS.llm:
            defn = AGENT_DEFS[agent_key]
            prior = llm_outputs.get(list(llm_outputs.keys())[-1], "") if llm_outputs else ""
            llm_outputs[agent_key] = _run_crewai_agent(
                role=defn["role"],
                goal=defn["goal"],
                backstory=defn.get("backstory", f"You are a {defn['role']}."),
                context_chunks=selected,
                task_description=f"Complete your task as {defn['role']}. Customer case: duplicate charge on Pro Plan order ORD-4492.",
                expected_output="Brief task result (2-3 sentences)",
                prior_task_result=prior,
            )
            print(f"    LLM → {llm_outputs[agent_key][:120]}...")

    print()

    # ── Phase 2: Fan-out — 5 parallel specialists ──────────────────────────
    _separator()
    print("  PHASE 2 — Fan-out: 5 Parallel Specialists  (Orchestrator dispatches)")
    _separator()

    fanout_agents = ["billing", "fraud", "compliance", "policy", "research"]
    fanout_results: dict[str, list[AnnotatedChunk]] = {}

    def _fanout_worker(agent_key: str):
        _, selected, tokens = run_agent(agent_key, SESSION, 2, "orchestrator", ledger, SESSION)
        llm_out = ""
        if ARGS.llm:
            defn = AGENT_DEFS[agent_key]
            llm_out = _run_crewai_agent(
                role=defn["role"],
                goal=defn["goal"],
                backstory=defn.get("backstory", f"You are a {defn['role']}."),
                context_chunks=selected,
                task_description=f"Analyze this customer case as {defn['role']}. Customer claims duplicate charge on Pro Plan, order ORD-4492.",
                expected_output="Your analysis and recommendation (2-3 sentences)",
            )
        return agent_key, selected, tokens, llm_out

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fanout_worker, key): key for key in fanout_agents}
        for future in as_completed(futures):
            agent_key, selected, tokens, llm_out = future.result()
            fanout_results[agent_key] = selected
            llm_outputs[agent_key] = llm_out
            _print_agent_result(agent_key, selected, 0, 2)
            if ARGS.llm and llm_out:
                print(f"    LLM → {llm_out[:120]}...")
            total_chp_tokens += tokens

    print()

    # ── Phase 3: Fan-in — orchestrator synthesizes ─────────────────────────
    _separator()
    print("  PHASE 3 — Fan-in: Orchestrator Synthesizes")
    _separator()

    # Subagent output chunks — use real LLM text if available, else summary stub
    subagent_outputs = []
    for agent_key, selected in fanout_results.items():
        defn = AGENT_DEFS[agent_key]
        manifest = infer_manifest(**defn)
        content = llm_outputs.get(agent_key) or (
            f"{defn['role']} result: selected {len(selected)} chunks, "
            f"{sum(c.token_cost for c in selected)} tokens"
        )
        subagent_outputs.append(AnnotatedChunk(
            chunk_id=f"out_{agent_key}",
            content=content[:200],   # cap to avoid token blowup
            token_cost=min(len(content.split()) * 2, 80),
            source_agent=manifest.agent_id,
            source_turn=3,
            timestamp=datetime.now(timezone.utc),
        ))

    orch_chunks = SESSION + subagent_outputs
    _, orch_selected, orch_tokens = run_agent("orchestrator", orch_chunks, 3, "all_specialists", ledger, orch_chunks)
    _print_agent_result("orchestrator", orch_selected, 0, 3)
    total_chp_tokens += orch_tokens

    if ARGS.llm:
        defn = AGENT_DEFS["orchestrator"]
        specialist_summaries = "\n".join(
            f"- {AGENT_DEFS[k]['role']}: {llm_outputs.get(k, 'no output')[:100]}"
            for k in fanout_agents
        )
        llm_outputs["orchestrator"] = _run_crewai_agent(
            role=defn["role"],
            goal=defn["goal"],
            backstory="You synthesize specialist reports and make the final call.",
            context_chunks=orch_selected,
            task_description=(
                "Make the final resolution decision for this customer case.\n\n"
                f"Specialist reports:\n{specialist_summaries}"
            ),
            expected_output="Final decision: approve/deny refund, reasoning, next steps (3-4 sentences)",
        )
        print(f"    LLM → {llm_outputs['orchestrator'][:200]}...")
    print()

    # ── Phase 4: Sequential resolution ────────────────────────────────────
    _separator()
    print("  PHASE 4 — Sequential Resolution  (Escalation → Summarizer)")
    _separator()

    seen_ids: set[str] = set()
    escalation_input: list[AnnotatedChunk] = []
    for c in SESSION + orch_selected:
        if c.chunk_id not in seen_ids:
            seen_ids.add(c.chunk_id)
            escalation_input.append(c)

    _, esc_selected, esc_tokens = run_agent("escalation", escalation_input, 4, "orchestrator", ledger, escalation_input)
    _print_agent_result("escalation", esc_selected, 0, 4)
    total_chp_tokens += esc_tokens

    if ARGS.llm:
        defn = AGENT_DEFS["escalation"]
        llm_outputs["escalation"] = _run_crewai_agent(
            role=defn["role"],
            goal=defn["goal"],
            backstory="You handle escalations and coordinate resolution for premium customers.",
            context_chunks=esc_selected,
            task_description="Confirm the resolution decision and determine if escalation is needed.",
            expected_output="Escalation decision and instructions for summarizer (2-3 sentences)",
            prior_task_result=llm_outputs.get("orchestrator", ""),
        )
        print(f"    LLM → {llm_outputs['escalation'][:120]}...")

    _, sum_selected, sum_tokens = run_agent("summarizer", esc_selected, 5, "escalation", ledger, esc_selected)
    _print_agent_result("summarizer", sum_selected, 0, 5)
    total_chp_tokens += sum_tokens

    if ARGS.llm:
        defn = AGENT_DEFS["summarizer"]
        llm_outputs["summarizer"] = _run_crewai_agent(
            role=defn["role"],
            goal=defn["goal"],
            backstory="You write clear, professional customer-facing resolution messages.",
            context_chunks=sum_selected,
            task_description="Write the final customer-facing message for this support case.",
            expected_output="Customer-facing resolution email (professional, 3-4 sentences, no PII)",
            prior_task_result=llm_outputs.get("escalation", ""),
        )
        print()
        _separator("─", 50)
        print("  FINAL CUSTOMER MESSAGE (LLM output):")
        _separator("─", 50)
        print(f"  {llm_outputs['summarizer']}")
        _separator("─", 50)
    print()

    # ── Background agents ──────────────────────────────────────────────────
    _separator()
    print("  BACKGROUND — Code Reviewer + Account Auditor  (async, non-blocking)")
    _separator()

    def _bg_worker(agent_key: str):
        _, selected, tokens = run_agent(agent_key, SESSION, 6, "router", ledger, SESSION)
        llm_out = ""
        if ARGS.llm:
            defn = AGENT_DEFS[agent_key]
            llm_out = _run_crewai_agent(
                role=defn["role"],
                goal=defn["goal"],
                backstory=defn.get("backstory", f"You are a {defn['role']}."),
                context_chunks=selected,
                task_description=f"Complete your background analysis as {defn['role']}.",
                expected_output="Background findings (2 sentences)",
            )
        return agent_key, selected, tokens, llm_out

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_bg_worker, key): key for key in ["code_reviewer", "auditor"]}
        for future in as_completed(futures):
            agent_key, selected, tokens, llm_out = future.result()
            _print_agent_result(agent_key, selected, 0, 6)
            if ARGS.llm and llm_out:
                print(f"    LLM → {llm_out[:120]}...")
            total_chp_tokens += tokens

    print()

    # ── Ledger stats + semantic query ──────────────────────────────────────
    _separator()
    print("  LEDGER STATS (Production)")
    _separator()
    stats = ledger.stats()
    print(f"  Ledger rows  : {stats['ledger_rows']}")
    print(f"  Chunk rows   : {stats['chunk_rows']}  (dedup: {len(SESSION)} session chunks stored once)")
    print()

    print("  Semantic query: 'billing refund duplicate charge'")
    sem_results = ledger.query_by_meaning(
        query_text="billing refund duplicate charge",
        embedder=embedder,
        session_id=SESSION_ID,
        top_k=3,
    )
    if sem_results:
        for r in sem_results:
            print(f"    → [{r.source_agent}] {r.content[:70]}")
    else:
        print("    (stub embedder returns uniform vectors — use --embeddings for ranked results)")
    print()

    # ── Session audit: hop-by-hop ─────────────────────────────────────────
    _separator()
    print("  LEDGER AUDIT — All hops")
    _separator()
    for hop_num in range(7):
        hop_records = ledger.query_hop(SESSION_ID, hop_num)
        if hop_records:
            agents = {r.hop_sequence[-1] if r.hop_sequence else "?" for r in hop_records}
            print(f"  Hop {hop_num}: {len(hop_records):3d} records  agents={sorted(agents)}")
    print()

    # ── Token summary ──────────────────────────────────────────────────────
    _separator("═")
    print("  TOKEN REDUCTION SUMMARY")
    _separator("═")
    saved    = total_baseline - total_chp_tokens
    pct      = saved / total_baseline * 100
    print(f"  Baseline (full context × {len(AGENT_DEFS)} agents) : {total_baseline:6d} tokens")
    print(f"  CHP (each agent gets only what it declared)  : {total_chp_tokens:6d} tokens")
    print(f"  Saved                                        : {saved:6d} tokens  ({pct:.1f}%)")
    print()

    # PII audit
    all_session_records = ledger.query(SESSION_ID)
    pii_leaks = [r for r in all_session_records if "PII_raw" in r.content or "credit_card" in r.content.lower()]
    non_compliance = [r for r in pii_leaks if "compliance" not in (r.hop_sequence[-1] if r.hop_sequence else "")]
    print(f"  PII_raw in ledger        : {len(pii_leaks)} records")
    print(f"  PII leaked to non-compliance agents : {len(non_compliance)}  ({'✓ SAFE' if not non_compliance else '✗ LEAK'})")
    print()

    # ── Prune + cleanup ────────────────────────────────────────────────────
    if not ARGS.no_prune:
        _separator()
        print("  LEDGER CLEANUP (TTL prune + orphan sweep + compaction)")
        _separator()
        pruned_rows  = ledger.prune(SESSION_ID)
        orphans      = ledger.prune_orphan_chunks()
        ledger.compact()
        print(f"  Pruned ledger rows  : {pruned_rows}")
        print(f"  Orphan chunks swept : {orphans}")
        print(f"  Final stats         : {ledger.stats()}")
        print()
    else:
        print(f"  --no-prune: ledger kept alive at session {SESSION_ID}")
        print()

    _separator("═")
    print("  Done.")
    _separator("═")
    print()


if __name__ == "__main__":
    main()
