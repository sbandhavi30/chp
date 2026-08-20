from __future__ import annotations
import json
import re
from chp.schema.context_manifest import ContextManifest, ContextRequirements

_INFER_PROMPT = """\
You are a context requirements analyzer for multi-agent AI systems.

Given an AI agent's role, goal, and backstory, infer a ContextManifest — \
a structured declaration of what context this agent needs to do its job.

Agent details:
  role      : {role}
  goal      : {goal}
  backstory : {backstory}

Pipeline agents available (may be empty):
  {pipeline_agents}

Return a JSON object with exactly these fields:
{{
  "must_carry": ["list of field names this agent MUST have to function — e.g. order_id, user_id"],
  "domain_tags": ["5-8 topic keywords this agent cares about — e.g. billing, refund, auth"],
  "history_depth": "one of: full | decisions_only | summary | none",
  "exclude": ["fields that should NEVER be passed to this agent — e.g. PII_raw, debug_trace, credit_card"],
  "token_budget": <integer, estimated tokens this agent needs: 200-5000>,
  "on_missing": "one of: fail_hard | warn | proceed | ledger_fallback",
  "accept_upstream_output": <true | false | ["agent-id-1", "agent-id-2"]>,
  "reasoning": "one sentence explaining the key decisions"
}}

Rules:
- must_carry: only fields the agent CANNOT work without
- exclude: always include PII_raw and debug_trace unless agent explicitly needs them (e.g. compliance)
- token_budget: small for focused agents (200-500), large for orchestrators/researchers (2000-5000)
- on_missing: fail_hard only if missing data would produce dangerous wrong output
- accept_upstream_output:
    * true  — agent synthesizes or depends on results from ALL prior agents (orchestrators, summarizers, escalation)
    * false — agent only needs pool context, not prior agent decisions (routers, auth, first-hop agents)
    * list  — agent depends on specific upstream agents named in pipeline_agents
              (e.g. fraud analyst needs billing result → ["billing-agent"])
    * If pipeline_agents is empty, use true or false only (cannot name specific agents)

Return ONLY the JSON object, no other text."""


def infer_manifest(
    role: str,
    goal: str,
    backstory: str = "",
    agent_id: str | None = None,
    llm_client=None,
    model: str = "gpt-4o-mini",
    pipeline_agents: list[str] | None = None,
) -> ContextManifest:
    """
    Infer a ContextManifest from an agent's role, goal, and backstory.

    Args:
        role: Agent's role string (e.g. "Billing Specialist")
        goal: Agent's goal string (e.g. "Resolve duplicate charge disputes")
        backstory: Optional backstory for richer inference
        agent_id: Optional agent ID override (defaults to slugified role)
        llm_client: OpenAI-compatible client. If None, uses heuristic inference.
        model: LLM model to use for inference
        pipeline_agents: Other agent IDs in this pipeline. When provided, the LLM
            can infer specific upstream dependencies (e.g. ["billing-agent"]) instead
            of just true/false. Enables fine-grained accept_upstream_output inference.

    Returns:
        ContextManifest ready to use or customise
    """
    inferred_id = agent_id or _slugify(role)

    if llm_client is not None:
        return _infer_with_llm(role, goal, backstory, inferred_id, llm_client, model, pipeline_agents)
    else:
        return _infer_heuristic(role, goal, backstory, inferred_id, pipeline_agents)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-based inference
# ─────────────────────────────────────────────────────────────────────────────

def _infer_with_llm(
    role: str,
    goal: str,
    backstory: str,
    agent_id: str,
    client,
    model: str,
    pipeline_agents: list[str] | None,
) -> ContextManifest:
    agents_str = ", ".join(pipeline_agents) if pipeline_agents else "(none provided)"
    prompt = _INFER_PROMPT.format(
        role=role,
        goal=goal,
        backstory=backstory or "not provided",
        pipeline_agents=agents_str,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return _infer_heuristic(role, goal, backstory, agent_id, pipeline_agents)

    must_carry    = [str(x) for x in data.get("must_carry", []) if isinstance(x, str)][:20]
    domain_tags   = [str(x) for x in data.get("domain_tags", []) if isinstance(x, str)][:20]
    exclude       = [str(x) for x in data.get("exclude", ["PII_raw", "debug_trace"]) if isinstance(x, str)][:20]
    history_depth = data.get("history_depth", "decisions_only")
    if history_depth not in ("full", "decisions_only", "summary", "none"):
        history_depth = "decisions_only"
    on_missing = data.get("on_missing", "warn")
    if on_missing not in ("fail_hard", "warn", "proceed", "ledger_fallback"):
        on_missing = "warn"
    token_budget = max(100, min(int(data.get("token_budget", 1000)), 50000))
    accept_upstream = _parse_accept_upstream(data.get("accept_upstream_output", False), pipeline_agents)

    return ContextManifest(
        agent_id=agent_id,
        task=_slugify(goal)[:60],
        requires=ContextRequirements(
            must_carry=must_carry,
            domain_tags=domain_tags,
            history_depth=history_depth,
            exclude=exclude,
            accept_upstream_output=accept_upstream,
        ),
        token_budget=token_budget,
        on_missing=on_missing,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic inference (no LLM — keyword matching)
# ─────────────────────────────────────────────────────────────────────────────

# Each rule: (trigger_keywords, overrides_dict)
# accept_upstream_output values:
#   False           — first-hop or standalone agents, need no upstream decisions
#   True            — synthesizers/orchestrators, need all upstream results
#   "__downstream__"— placeholder: inferred from goal keywords at runtime
#     (replaced with specific agent IDs when pipeline_agents provided)
_KEYWORD_RULES: list[tuple[list[str], dict]] = [
    (
        ["billing", "charge", "payment", "refund", "invoice", "duplicate"],
        {"must_carry": ["order_id", "user_id"],
         "domain_tags": ["billing", "order", "charge", "refund", "payment"],
         "on_missing": "fail_hard", "token_budget": 1500,
         # billing needs auth result to confirm identity before approving refund
         "accept_upstream_output": ["auth"],
         "upstream_keywords": ["auth"]},
    ),
    (
        ["auth", "identity", "verify", "login", "session", "mfa"],
        {"must_carry": ["user_id", "auth_status"],
         "domain_tags": ["auth", "identity", "session", "verification"],
         "on_missing": "warn", "token_budget": 500,
         # auth is typically first-hop after router — no upstream decisions needed
         "accept_upstream_output": False},
    ),
    (
        ["fraud", "risk", "anomaly", "suspicious", "velocity"],
        {"must_carry": ["fraud_score"],
         "domain_tags": ["fraud", "risk", "velocity", "device", "anomaly"],
         "on_missing": "warn", "token_budget": 600,
         # fraud needs billing decision to correlate charge pattern
         "accept_upstream_output": ["billing"],
         "upstream_keywords": ["billing"]},
    ),
    (
        ["compliance", "gdpr", "pci", "regulatory", "legal", "audit"],
        {"must_carry": ["user_id", "compliance_flags"],
         "domain_tags": ["compliance", "GDPR", "PCI", "regulatory"],
         "on_missing": "fail_hard", "token_budget": 800,
         "exclude": ["debug_trace"],
         # compliance needs auth + billing decisions
         "accept_upstream_output": ["auth", "billing"],
         "upstream_keywords": ["auth", "billing"]},
    ),
    (
        ["summar", "resolution", "customer-facing", "response", "conclude"],
        {"must_carry": [],
         "domain_tags": ["summary", "resolution", "outcome", "customer_request"],
         "on_missing": "warn", "token_budget": 1000,
         "exclude_extra": ["auth_status", "session_token", "fraud_score"],
         # summarizer needs orchestrator + escalation final decision
         "accept_upstream_output": ["orchestrator", "escalation"],
         "upstream_keywords": ["orchestrat", "escalat"]},
    ),
    (
        ["research", "retrieve", "search", "find", "investigate"],
        {"must_carry": [],
         "domain_tags": ["research", "facts", "knowledge", "sources"],
         "on_missing": "proceed", "token_budget": 3000,
         # research is standalone — works from pool context only
         "accept_upstream_output": False},
    ),
    (
        ["orchestrat", "coordinat", "synthesize", "delegate", "plan"],
        {"must_carry": [],
         "domain_tags": ["summary", "result", "decision", "outcome"],
         "on_missing": "proceed", "token_budget": 4000,
         # orchestrator synthesizes ALL specialist results
         "accept_upstream_output": True},
    ),
    (
        ["policy", "eligib", "terms", "rules", "entitl"],
        {"must_carry": ["order_id"],
         "domain_tags": ["policy", "eligibility", "terms", "rules"],
         "on_missing": "fail_hard", "token_budget": 800,
         # policy only needs pool context (order details), not prior agent decisions
         "accept_upstream_output": False},
    ),
    (
        ["code", "review", "diff", "pull request", "security", "lint"],
        {"must_carry": [],
         "domain_tags": ["code", "diff", "review", "security", "style"],
         "on_missing": "proceed", "token_budget": 5000,
         # code reviewer is standalone
         "accept_upstream_output": False},
    ),
    (
        ["route", "triage", "classif", "dispatch", "intent"],
        {"must_carry": ["user_id"],
         "domain_tags": ["routing", "classification", "intent", "triage"],
         "on_missing": "warn", "token_budget": 400,
         # router is first-hop — nothing upstream
         "accept_upstream_output": False},
    ),
    (
        ["escalat", "senior", "manag", "escalation"],
        {"must_carry": [],
         "domain_tags": ["escalation", "management", "senior", "decision"],
         "on_missing": "warn", "token_budget": 1200,
         # escalation needs orchestrator synthesis
         "accept_upstream_output": ["orchestrator"],
         "upstream_keywords": ["orchestrat"]},
    ),
]

_DEFAULT_EXCLUDE = ["PII_raw", "credit_card", "ssn", "dob", "debug_trace", "stack_trace"]


def _infer_heuristic(
    role: str,
    goal: str,
    backstory: str,
    agent_id: str,
    pipeline_agents: list[str] | None = None,
) -> ContextManifest:
    text = f"{role} {goal} {backstory}".lower()

    best_match: dict | None = None
    best_score = 0

    for keywords, overrides in _KEYWORD_RULES:
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_match = overrides

    if best_match is None:
        best_match = {
            "must_carry": [],
            "domain_tags": _extract_keywords(text),
            "on_missing": "warn",
            "token_budget": 1000,
            "accept_upstream_output": False,
        }

    exclude = best_match.get("exclude", list(_DEFAULT_EXCLUDE))
    exclude += best_match.get("exclude_extra", [])

    raw_accept = best_match.get("accept_upstream_output", False)
    accept_upstream = _resolve_upstream_agents(raw_accept, best_match.get("upstream_keywords", []), pipeline_agents)

    return ContextManifest(
        agent_id=agent_id,
        task=_slugify(goal)[:60],
        requires=ContextRequirements(
            must_carry=best_match.get("must_carry", []),
            domain_tags=best_match.get("domain_tags", []),
            history_depth="decisions_only",
            exclude=list(set(exclude)),
            accept_upstream_output=accept_upstream,
        ),
        token_budget=best_match.get("token_budget", 1000),
        on_missing=best_match.get("on_missing", "warn"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_accept_upstream(raw, pipeline_agents: list[str] | None) -> bool | list[str]:
    """
    Parse and validate the LLM-returned accept_upstream_output value.

    - bool → pass through
    - list of strings → filter to only IDs present in pipeline_agents (if provided)
    - anything else → False (safe default)
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, list):
        ids = [str(x) for x in raw if isinstance(x, str)][:20]
        if not ids:
            return False
        # If pipeline_agents known, only keep IDs that actually exist in pipeline
        if pipeline_agents:
            ids = [i for i in ids if i in pipeline_agents]
        return ids if ids else False
    return False


def _resolve_upstream_agents(
    raw_accept,
    upstream_keywords: list[str],
    pipeline_agents: list[str] | None,
) -> bool | list[str]:
    """
    For heuristic rules: raw_accept is already True/False/list[str].
    When raw_accept is a list of keyword fragments (e.g. ["auth", "billing"]),
    match them against actual pipeline_agents IDs if provided.
    """
    if isinstance(raw_accept, bool):
        return raw_accept

    if isinstance(raw_accept, list):
        keyword_fragments = raw_accept  # e.g. ["auth", "billing", "orchestrat"]
        if not pipeline_agents:
            # No pipeline context — fall back to True if any upstream expected, else False
            return bool(keyword_fragments)
        # Match fragment against pipeline agent IDs
        matched = [
            agent_id for agent_id in pipeline_agents
            if any(frag.lower() in agent_id.lower() for frag in keyword_fragments)
        ]
        return matched if matched else False

    return False


def _extract_keywords(text: str) -> list[str]:
    stopwords = {"the", "a", "an", "is", "are", "to", "for", "of", "and", "or",
                 "in", "on", "at", "by", "with", "this", "that", "you", "your"}
    words = re.findall(r'\b[a-z]{4,}\b', text)
    seen: dict[str, int] = {}
    for w in words:
        if w not in stopwords:
            seen[w] = seen.get(w, 0) + 1
    return [w for w, _ in sorted(seen.items(), key=lambda x: -x[1])][:6]


def _slugify(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')
