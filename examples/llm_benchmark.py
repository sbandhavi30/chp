"""
CHP Real LLM Benchmark — Before vs After, 12 Agents, 3 Topologies
===================================================================
Runs the full 12-agent customer support pipeline twice:
  1. BASELINE — each agent gets full 14-chunk pool (no filtering)
  2. CHP      — each agent gets only what its manifest declares

Pipeline topology mirrors full_pipeline_demo.py:
  Phase 1 — Sequential triage:  Router → Auth
  Phase 2 — Fan-out (parallel): Billing, Fraud, Compliance, Policy, Research
  Phase 3 — Fan-in:             Orchestrator synthesizes all 5
  Phase 4 — Sequential:         Escalation → Summarizer
  Background (parallel):        Code Reviewer, Account Auditor

Measures per run:
  • Actual prompt tokens per agent (via tiktoken or word-count estimate)
  • Completion tokens
  • Wall-clock latency per agent
  • PII present in prompt (yes/no per agent)
  • Final billing decision (APPROVED / DENIED / UNCLEAR)
  • Token reduction vs baseline

Requirements:
    export OPENAI_API_KEY=sk-...
    pip install crewai openai chp
    pip install tiktoken   # optional, more accurate token counts

Run:
    python examples/llm_benchmark.py
    python examples/llm_benchmark.py --model gpt-4o-mini   # default, cheapest
    python examples/llm_benchmark.py --model gpt-4o
    python examples/llm_benchmark.py --json results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field

# ── Validate environment ──────────────────────────────────────────────────────

if not os.environ.get("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not set.")
    print("  export OPENAI_API_KEY=sk-...")
    sys.exit(1)

try:
    from crewai import Agent, Task, Crew, Process as CrewProcess
except ImportError:
    print("ERROR: crewai not installed.  pip install crewai")
    sys.exit(1)

# ── CHP path ──────────────────────────────────────────────────────────────────

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from chp.schema.rationale_envelope import AnnotatedChunk
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.engine.embedder import StubEmbedder
from chp.engine.scorer import select_chunks
from chp.pii import RegexPIIFilter
from chp.observability import SessionTokenTracker
import chp

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="CHP 12-agent real LLM benchmark")
parser.add_argument("--model",  default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
parser.add_argument("--json",   metavar="FILE",        help="Write full results to JSON file")
ARGS = parser.parse_args()

# ── Token counting ────────────────────────────────────────────────────────────

try:
    import tiktoken
    _enc = tiktoken.encoding_for_model("gpt-4o-mini")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return int(len(text.split()) * 1.3)

# ── PII detection ─────────────────────────────────────────────────────────────

_pii = RegexPIIFilter(log_detections=False)

# ── Context pool — 14 chunks ─────────────────────────────────────────────────

CHUNKS: list[AnnotatedChunk] = [
    AnnotatedChunk(chunk_id="c_user",       content="user_id: USR-8821 | tier: premium | account_age: 3yr | status: active",                           token_cost=30, source_agent="router", source_turn=1),
    AnnotatedChunk(chunk_id="c_order",      content="order_id: ORD-4492 | amount: $299 | date: 2026-08-01 | plan: Pro | status: charged_twice",         token_cost=35, source_agent="router", source_turn=2),
    AnnotatedChunk(chunk_id="c_pii",        content="PII_raw: credit_card=4111111111111111 | ssn=123-45-6789 | dob=1985-03-12",                         token_cost=40, source_agent="router", source_turn=3),
    AnnotatedChunk(chunk_id="c_auth",       content="auth_status: verified | mfa: passed | last_login: 2026-08-01T09:12Z | risk: low",                  token_cost=35, source_agent="router", source_turn=4),
    AnnotatedChunk(chunk_id="c_fraud",      content="fraud_score: 0.12 | ip_risk: low | device: known | velocity: normal | flag: none",                  token_cost=40, source_agent="router", source_turn=5),
    AnnotatedChunk(chunk_id="c_compliance", content="compliance_flags: GDPR=ok | PCI=flagged_raw_card_in_context | SOX=ok | residency=EU",               token_cost=40, source_agent="router", source_turn=6),
    AnnotatedChunk(chunk_id="c_policy",     content="refund_policy: Pro Plan 30-day full refund eligible | duplicate_charge: auto-approve under $500",   token_cost=45, source_agent="router", source_turn=7),
    AnnotatedChunk(chunk_id="c_request",    content="customer_request: charged twice for Pro Plan Aug 1, order ORD-4492, requesting refund of $299",     token_cost=45, source_agent="router", source_turn=8),
    AnnotatedChunk(chunk_id="c_debug",      content="debug_trace: POST /api/charge 200 | idempotency_key=missing | charge_id=ch_001 ch_002 duplicate",   token_cost=55, source_agent="router", source_turn=9),
    AnnotatedChunk(chunk_id="c_history",    content="prior_tickets: TKT-221 refund $50 approved 2026-07-15 | TKT-198 billing resolved 2026-05",          token_cost=45, source_agent="router", source_turn=10),
    AnnotatedChunk(chunk_id="c_diff",       content="git_diff: billing_service.py idempotency fix +charge_v2 -charge_v1 unrelated code change",          token_cost=50, source_agent="router", source_turn=11),
    AnnotatedChunk(chunk_id="c_kb",         content="knowledge_base: duplicate charge SOP — verify idempotency key, check charge IDs, auto-approve <$500", token_cost=55, source_agent="router", source_turn=12),
    AnnotatedChunk(chunk_id="c_escalation", content="escalation_rules: premium SLA 2hr | duplicate_charge: auto-approve no manager needed if <$500",     token_cost=40, source_agent="router", source_turn=13),
    AnnotatedChunk(chunk_id="c_outcome_tpl",content="outcome_template: Dear {name}, your refund of {amount} has been {status}. Reference: {order_id}",   token_cost=35, source_agent="router", source_turn=14),
]

POOL_TOKENS = sum(c.token_cost for c in CHUNKS)

# ── Manifests — one per agent ─────────────────────────────────────────────────

_PII_EXCLUDE = ["ssn", "credit_card", "PII_raw", "dob"]

MANIFESTS: dict[str, ContextManifest] = {
    "router": ContextManifest(agent_id="router", task="Classify and route support ticket",
        requires=ContextRequirements(must_carry=["customer_request", "order_id"],
            domain_tags=["customer", "request", "order"], exclude=_PII_EXCLUDE), token_budget=200),
    "auth": ContextManifest(agent_id="auth-agent", task="Verify customer identity",
        requires=ContextRequirements(must_carry=["auth_status", "user_id"],
            domain_tags=["auth", "identity", "verification"], exclude=_PII_EXCLUDE), token_budget=150),
    "billing": ContextManifest(agent_id="billing-agent", task="Resolve duplicate charge and approve refund",
        requires=ContextRequirements(must_carry=["order_id", "refund_policy"],
            domain_tags=["billing", "refund", "duplicate", "policy"], exclude=_PII_EXCLUDE + ["git_diff"]), token_budget=250),
    "fraud": ContextManifest(agent_id="fraud-agent", task="Assess fraud risk for this transaction",
        requires=ContextRequirements(must_carry=["fraud_score"],
            domain_tags=["fraud", "risk", "velocity", "device"], exclude=_PII_EXCLUDE), token_budget=150),
    "compliance": ContextManifest(agent_id="compliance-agent", task="Check GDPR and PCI compliance",
        requires=ContextRequirements(must_carry=["compliance_flags"],
            domain_tags=["compliance", "GDPR", "PCI"], exclude=[]), token_budget=150),
    "policy": ContextManifest(agent_id="policy-agent", task="Check refund policy eligibility",
        requires=ContextRequirements(must_carry=["refund_policy", "order_id"],
            domain_tags=["policy", "refund", "eligibility"], exclude=_PII_EXCLUDE), token_budget=150),
    "research": ContextManifest(agent_id="research-agent", task="Retrieve knowledge base articles for duplicate charge",
        requires=ContextRequirements(must_carry=["knowledge_base"],
            domain_tags=["knowledge", "SOP", "duplicate", "resolution"], exclude=_PII_EXCLUDE), token_budget=150),
    "orchestrator": ContextManifest(agent_id="orchestrator", task="Synthesize specialist results and make final decision",
        requires=ContextRequirements(must_carry=["order_id"],
            domain_tags=["billing", "fraud", "policy", "compliance", "resolution"], exclude=_PII_EXCLUDE), token_budget=400),
    "escalation": ContextManifest(agent_id="escalation-agent", task="Confirm resolution and check if escalation needed",
        requires=ContextRequirements(must_carry=["escalation_rules", "order_id"],
            domain_tags=["escalation", "SLA", "premium", "resolution"], exclude=_PII_EXCLUDE), token_budget=200),
    "summarizer": ContextManifest(agent_id="summarizer", task="Write customer-facing resolution message",
        requires=ContextRequirements(must_carry=["order_id", "outcome_template"],
            domain_tags=["outcome", "refund", "customer", "resolution"], exclude=_PII_EXCLUDE + ["debug_trace", "git_diff"]), token_budget=150),
    "code_reviewer": ContextManifest(agent_id="code-reviewer", task="Review billing service diff for security issues",
        requires=ContextRequirements(must_carry=["git_diff"],
            domain_tags=["git", "code", "security", "billing"], exclude=_PII_EXCLUDE), token_budget=150),
    "auditor": ContextManifest(agent_id="auditor", task="Audit account history for suspicious patterns",
        requires=ContextRequirements(must_carry=["user_id", "prior_tickets"],
            domain_tags=["audit", "account", "history", "identity"], exclude=_PII_EXCLUDE), token_budget=150),
}

# ── Agent task definitions ────────────────────────────────────────────────────

AGENT_DEFS: dict[str, dict] = {
    "router":       {"role": "Support Router",      "goal": "Classify and route support ticket to billing",            "backstory": "You triage incoming tickets.", "task": "Confirm this is a duplicate charge case for order ORD-4492 and route to billing.", "expected_output": "Routing decision (1-2 sentences)"},
    "auth":         {"role": "Auth Specialist",     "goal": "Verify customer identity",                                "backstory": "You confirm auth before billing acts.", "task": "Confirm customer identity is verified and safe to proceed.", "expected_output": "Auth clearance (1-2 sentences)"},
    "billing":      {"role": "Billing Specialist",  "goal": "Resolve duplicate charge and approve refund under $500", "backstory": "You have authority to approve refunds under $500.", "task": "Review duplicate charge for ORD-4492 ($299). State APPROVED or DENIED.", "expected_output": "Decision: APPROVED or DENIED with reason (2-3 sentences)"},
    "fraud":        {"role": "Fraud Analyst",       "goal": "Assess fraud risk for this transaction",                 "backstory": "You flag suspicious transactions.", "task": "Assess the fraud risk for this duplicate charge case.", "expected_output": "Fraud risk assessment (1-2 sentences)"},
    "compliance":   {"role": "Compliance Officer",  "goal": "Check GDPR and PCI compliance",                          "backstory": "You ensure data handling meets regulations.", "task": "Check if this case has any GDPR or PCI compliance issues.", "expected_output": "Compliance status (1-2 sentences)"},
    "policy":       {"role": "Policy Checker",      "goal": "Verify refund policy eligibility",                       "backstory": "You enforce refund policies.", "task": "Confirm if ORD-4492 qualifies for a refund under policy.", "expected_output": "Policy eligibility (1-2 sentences)"},
    "research":     {"role": "Research Agent",      "goal": "Find relevant SOP for duplicate charge resolution",      "backstory": "You retrieve knowledge base articles.", "task": "Find the SOP for handling duplicate charge disputes.", "expected_output": "Relevant SOP steps (2-3 sentences)"},
    "orchestrator": {"role": "Orchestrator",        "goal": "Synthesize specialist results and make final call",      "backstory": "You make the final resolution decision.", "task": "Given specialist inputs, make the final refund decision for ORD-4492.", "expected_output": "Final decision with reasoning (3-4 sentences)"},
    "escalation":   {"role": "Escalation Manager",  "goal": "Confirm resolution, check if escalation needed",         "backstory": "You handle escalations for premium customers.", "task": "Confirm the resolution decision and determine if escalation is needed.", "expected_output": "Escalation decision (1-2 sentences)"},
    "summarizer":   {"role": "Customer Summarizer", "goal": "Write professional customer-facing resolution message",  "backstory": "You write clear messages. Never include PII or internal IDs.", "task": "Write a customer email confirming resolution for ORD-4492. No PII.", "expected_output": "Customer email (2-3 sentences, no PII)"},
    "code_reviewer":{"role": "Code Reviewer",       "goal": "Review billing service diff for security issues",        "backstory": "You catch security issues in code changes.", "task": "Review the billing service diff and flag any security issues.", "expected_output": "Security review (2 sentences)"},
    "auditor":      {"role": "Account Auditor",     "goal": "Audit account history for suspicious patterns",          "backstory": "You verify account integrity.", "task": "Review account history and flag any suspicious patterns.", "expected_output": "Audit findings (2 sentences)"},
}

# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class AgentRun:
    agent_id: str
    method: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    pii_in_prompt: bool
    chunk_count: int
    chunk_ids: list[str]
    output: str


@dataclass
class PipelineRun:
    method: str
    model: str
    agent_runs: list[AgentRun] = field(default_factory=list)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.agent_runs)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.agent_runs)

    @property
    def total_latency_ms(self) -> float:
        return sum(r.latency_ms for r in self.agent_runs)

    @property
    def pii_leaked(self) -> bool:
        return any(r.pii_in_prompt for r in self.agent_runs)

    @property
    def pii_agents(self) -> list[str]:
        return [r.agent_id for r in self.agent_runs if r.pii_in_prompt]

    def decision(self) -> str:
        for r in self.agent_runs:
            if r.agent_id in ("billing", "orchestrator"):
                out = r.output.upper()
                if "APPROVED" in out: return "APPROVED"
                if "DENIED"   in out: return "DENIED"
        return "UNCLEAR"


# ── Core: run one agent against LLM ──────────────────────────────────────────

def _run_agent(
    agent_id: str,
    context_chunks: list[AnnotatedChunk],
    method: str,
    prior_output: str = "",
) -> AgentRun:
    defn = AGENT_DEFS[agent_id]
    ctx  = "\n".join(f"[{c.chunk_id}] {c.content}" for c in context_chunks)
    pii_in_prompt = _pii.contains_pii(ctx)

    task_desc = defn["task"]
    if prior_output:
        task_desc += f"\n\nPrior output:\n{prior_output[:400]}"
    task_desc += f"\n\nContext (CHP-filtered):\n{ctx}"

    agent = Agent(role=defn["role"], goal=defn["goal"], backstory=defn["backstory"],
                  verbose=False, llm=f"openai/{ARGS.model}")
    task  = Task(description=task_desc, agent=agent, expected_output=defn["expected_output"])
    crew  = Crew(agents=[agent], tasks=[task], process=CrewProcess.sequential, verbose=False)

    t0 = time.perf_counter()
    result = crew.kickoff()
    latency_ms = (time.perf_counter() - t0) * 1000

    output_str = str(result)
    return AgentRun(
        agent_id=agent_id,
        method=method,
        prompt_tokens=_count_tokens(task_desc),
        completion_tokens=_count_tokens(output_str),
        latency_ms=round(latency_ms, 0),
        pii_in_prompt=pii_in_prompt,
        chunk_count=len(context_chunks),
        chunk_ids=[c.chunk_id for c in context_chunks],
        output=output_str[:400],
    )


def _context(agent_id: str, method: str, extra_chunks: list[AnnotatedChunk] | None = None) -> list[AnnotatedChunk]:
    pool = CHUNKS + (extra_chunks or [])
    if method == "baseline":
        return pool
    return select_chunks(pool, MANIFESTS[agent_id], StubEmbedder())


def _log(r: AgentRun) -> None:
    pii = "PII:YES" if r.pii_in_prompt else "PII:no "
    print(
        f"    [{r.agent_id:<14}] chunks={r.chunk_count:2d}  "
        f"prompt_tokens={r.prompt_tokens:5,}  {pii}  {r.latency_ms/1000:.1f}s"
    )


# ── Full 12-agent pipeline ────────────────────────────────────────────────────

def run_pipeline(method: str) -> PipelineRun:
    run = PipelineRun(method=method, model=ARGS.model)
    outputs: dict[str, str] = {}

    # Phase 1 — Sequential triage
    print(f"\n  Phase 1 — Sequential triage (Router → Auth)")
    for agent_id, prior_key in [("router", None), ("auth", "router")]:
        prior = outputs.get(prior_key, "") if prior_key else ""
        r = _run_agent(agent_id, _context(agent_id, method), method, prior)
        run.agent_runs.append(r)
        outputs[agent_id] = r.output
        _log(r)

    # Phase 2 — Fan-out: 5 parallel specialists
    print(f"\n  Phase 2 — Fan-out: Billing, Fraud, Compliance, Policy, Research (parallel)")
    fanout = ["billing", "fraud", "compliance", "policy", "research"]

    def _fanout(agent_id: str):
        r = _run_agent(agent_id, _context(agent_id, method), method)
        return agent_id, r

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_fanout, a): a for a in fanout}
        for future in as_completed(futures):
            agent_id, r = future.result()
            run.agent_runs.append(r)
            outputs[agent_id] = r.output
            _log(r)

    # Phase 3 — Fan-in: Orchestrator
    print(f"\n  Phase 3 — Fan-in: Orchestrator")
    specialist_summary = "\n".join(
        f"- {AGENT_DEFS[a]['role']}: {outputs.get(a, 'no output')[:120]}"
        for a in fanout
    )
    # Synthesized subagent outputs injected as extra chunks
    subagent_chunks = [
        AnnotatedChunk(
            chunk_id=f"out_{a}", content=f"{AGENT_DEFS[a]['role']} result: {outputs.get(a,'')[:150]}",
            token_cost=40, source_agent=a, source_turn=3,
        )
        for a in fanout
    ]
    orch_r = _run_agent("orchestrator", _context("orchestrator", method, subagent_chunks), method, specialist_summary)
    run.agent_runs.append(orch_r)
    outputs["orchestrator"] = orch_r.output
    _log(orch_r)

    # Phase 4 — Sequential resolution
    print(f"\n  Phase 4 — Sequential resolution (Escalation → Summarizer)")
    esc_r = _run_agent("escalation", _context("escalation", method), method, outputs.get("orchestrator", ""))
    run.agent_runs.append(esc_r)
    outputs["escalation"] = esc_r.output
    _log(esc_r)

    sum_r = _run_agent("summarizer", _context("summarizer", method), method, outputs.get("escalation", ""))
    run.agent_runs.append(sum_r)
    outputs["summarizer"] = sum_r.output
    _log(sum_r)

    # Background agents (parallel)
    print(f"\n  Background — Code Reviewer + Account Auditor (parallel)")
    bg = ["code_reviewer", "auditor"]

    def _bg(agent_id: str):
        r = _run_agent(agent_id, _context(agent_id, method), method)
        return agent_id, r

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_bg, a): a for a in bg}
        for future in as_completed(futures):
            agent_id, r = future.result()
            run.agent_runs.append(r)
            outputs[agent_id] = r.output
            _log(r)

    return run


# ── Print results ─────────────────────────────────────────────────────────────

def print_results(b: PipelineRun, c: PipelineRun) -> None:
    reduction = (b.total_prompt_tokens - c.total_prompt_tokens) / b.total_prompt_tokens * 100
    print()
    print("=" * 76)
    print("  CHP vs Baseline — 12-Agent Real LLM Benchmark")
    print("=" * 76)
    print(f"  Model:    {ARGS.model}")
    print(f"  Agents:   12  |  Topologies: sequential + fan-out + fan-in + background")
    print(f"  Case:     Duplicate charge ORD-4492 $299 Pro Plan")
    print()
    print(f"  {'Metric':<42} {'Baseline':>12} {'CHP':>12}")
    print(f"  {'-'*42} {'-'*12} {'-'*12}")
    print(f"  {'Total prompt tokens (all 12 agents)':<42} {b.total_prompt_tokens:>12,} {c.total_prompt_tokens:>12,}")
    print(f"  {'Prompt token reduction':<42} {'—':>12} {reduction:>11.1f}%")
    print(f"  {'Total completion tokens':<42} {b.total_completion_tokens:>12,} {c.total_completion_tokens:>12,}")
    print(f"  {'PII reached any agent':<42} {'YES' if b.pii_leaked else 'NO':>12} {'YES' if c.pii_leaked else 'NO':>12}")
    print(f"  {'Agents that saw PII':<42} {', '.join(b.pii_agents) or 'none':>12} {', '.join(c.pii_agents) or 'none':>12}")
    print(f"  {'Final decision':<42} {b.decision():>12} {c.decision():>12}")
    print(f"  {'Total wall-clock latency':<42} {b.total_latency_ms/1000:>11.1f}s {c.total_latency_ms/1000:>11.1f}s")
    print()

    # Per-agent table
    b_map = {r.agent_id: r for r in b.agent_runs}
    c_map = {r.agent_id: r for r in c.agent_runs}
    all_ids = [r.agent_id for r in b.agent_runs]

    print(f"  {'Agent':<16} {'Base chunks':>12} {'CHP chunks':>11} {'Base tokens':>12} {'CHP tokens':>11} {'Reduction':>10} {'PII base':>9} {'PII chp':>8}")
    print(f"  {'-'*16} {'-'*12} {'-'*11} {'-'*12} {'-'*11} {'-'*10} {'-'*9} {'-'*8}")
    for aid in all_ids:
        br = b_map.get(aid)
        cr = c_map.get(aid)
        if not br or not cr:
            continue
        red = (br.prompt_tokens - cr.prompt_tokens) / br.prompt_tokens * 100
        print(
            f"  {aid:<16} {br.chunk_count:>12} {cr.chunk_count:>11} "
            f"{br.prompt_tokens:>12,} {cr.prompt_tokens:>11,} "
            f"{red:>9.1f}% {'YES' if br.pii_in_prompt else 'no':>9} {'YES' if cr.pii_in_prompt else 'no':>8}"
        )
    print()

    # Show key LLM outputs
    for agent_id, label in [("billing", "BILLING DECISION"), ("orchestrator", "ORCHESTRATOR"), ("summarizer", "CUSTOMER MESSAGE")]:
        br = b_map.get(agent_id)
        cr = c_map.get(agent_id)
        if br and cr:
            print(f"  {label}:")
            print(f"    Baseline: {br.output[:200]}")
            print(f"    CHP:      {cr.output[:200]}")
            print()

    print("=" * 76)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("=" * 76)
    print(f"  CHP 12-Agent Real LLM Benchmark  |  model={ARGS.model}")
    print("=" * 76)
    print(f"  Pool: {len(CHUNKS)} chunks, {POOL_TOKENS} tokens")
    print(f"  Baseline: each agent receives all {len(CHUNKS)} chunks")
    print(f"  CHP:      each agent receives only manifest-declared chunks")
    print()

    print("─" * 76)
    print("  BASELINE RUN")
    print("─" * 76)
    baseline = run_pipeline("baseline")

    print()
    print("─" * 76)
    print("  CHP RUN")
    print("─" * 76)
    chp_run = run_pipeline("chp")

    print_results(baseline, chp_run)

    if ARGS.json:
        out = {
            "model": ARGS.model,
            "pool_chunks": len(CHUNKS),
            "pool_tokens": POOL_TOKENS,
            "baseline": asdict(baseline),
            "chp": asdict(chp_run),
        }
        with open(ARGS.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  Results written to {ARGS.json}")


if __name__ == "__main__":
    main()
