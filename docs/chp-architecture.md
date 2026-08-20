# CHP Architecture

## System Overview

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     CONTEXT HANDOFF PROTOCOL (CHP)                             ║
║              The missing coordination layer for multi-agent AI                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝

  PROBLEM TODAY                         WITH CHP
  ─────────────                         ────────
  Agent A ──► [everything] ──► Agent B  Agent A ──► [B's manifest] ──► Agent B
              leaks PII                             only what B needs
              exceeds budget                        PII blocked at source
              no audit trail                        full provenance record


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  THE 4-LAYER STACK
  ─────────────────

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  LAYER 4 │ ContextLedger                                                   │
  │          │  append-only · two-table LanceDB · content-deduped · TTL        │
  │          │  chp_chunks (content + embeddings)  chp_ledger (provenance)     │
  ├──────────┼──────────────────────────────────────────────────────────────────┤
  │  LAYER 3 │ RationaleEnvelope                                               │
  │          │  per-chunk provenance · hop_sequence · selected_because         │
  │          │  score · must_carry · token_cost · ledger_id                    │
  ├──────────┼──────────────────────────────────────────────────────────────────┤
  │  LAYER 2 │ SelectionEngine (scorer)                                        │
  │          │  score = α·sim + β·recency + γ·must_carry - δ·cost - ε·exclude │
  │          │  weights: α=0.4  β=0.2  γ=0.3  δ=0.05  ε=0.05                 │
  ├──────────┼──────────────────────────────────────────────────────────────────┤
  │  LAYER 1 │ ContextManifest                                                 │
  │          │  must_carry · domain_tags · history_depth · exclude             │
  │          │  token_budget · on_missing · agent declares its OWN needs       │
  └──────────┴──────────────────────────────────────────────────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  CONTEXT FLOW (per hop)
  ──────────────────────

  Agent B declares manifest          Agent A selects from pool
  ┌───────────────────────┐          ┌──────────────────────────┐
  │ ContextManifest       │          │ Context Pool (all chunks) │
  │  must_carry: order_id │──────►   │  order_id ✓ must_carry   │
  │  domain_tags: billing │  scorer  │  user_id  ✓ domain match │
  │  exclude: PII_raw     │          │  PII_raw  ✗ hard block   │
  │  token_budget: 1500   │          │  debug    ✗ over budget  │
  └───────────────────────┘          └──────────────────────────┘
                                               │
                                               ▼
                                      RationaleEnvelope ×N
                                      (scored, annotated)
                                               │
                                    ┌──────────▼──────────┐
                                    │   ContextLedger      │
                                    │   write(session,     │
                                    │     hop, from, to,   │
                                    │     envelope)        │
                                    └─────────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MULTI-AGENT TOPOLOGIES
  ──────────────────────

  SEQUENTIAL CHAIN
  ─────────────────
  ┌────────┐    ┌──────┐    ┌─────────┐    ┌───────────┐
  │ Router │───►│ Auth │───►│ Billing │───►│ Summarizer│
  └────────┘    └──────┘    └─────────┘    └───────────┘
  Each hop: manifest-filtered · PII excluded · provenance recorded

  FAN-OUT (parallel subagents)
  ─────────────────────────────
                    ┌──────────┐
               ┌───►│ Billing  │
               │    └──────────┘
               │    ┌──────────┐
  ┌───────────┐├───►│ Fraud    │
  │Orchestrator│    └──────────┘   Each subagent gets
  │           │├───►│Compliance│   its OWN CHP-filtered
  └───────────┘│    └──────────┘   context independently
               │    ┌──────────┐
               └───►│ Research │
                    └──────────┘

  FAN-IN (synthesis)
  ───────────────────
  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Billing  │ │ Fraud    │ │Compliance│ │ Policy   │
  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
       │             │             │             │
       └─────────────┴─────────────┴─────────────┘
                              │ ledger.query(session)
                              ▼
                    ┌──────────────────┐
                    │  Orchestrator    │  full audit trail
                    │  + Synthesizer   │  recoverable at any hop
                    └──────────────────┘


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  LANCEDB TWO-TABLE DESIGN
  ─────────────────────────

  chp_chunks (content store — deduped)
  ┌──────────────┬───────────┬──────────────┬────────────┬──────────────────────┐
  │  chunk_id    │  content  │ source_agent │ token_cost │  embedding [384d]    │
  ├──────────────┼───────────┼──────────────┼────────────┼──────────────────────┤
  │  c_order_42  │ ORD-42... │   router     │     35     │  [0.12, -0.44, ...]  │
  │  c_user_001  │ USR-001...│   router     │     28     │  [0.33,  0.11, ...]  │
  └──────────────┴───────────┴──────────────┴────────────┴──────────────────────┘
        ▲ stored ONCE even if 10 agents select same chunk

  chp_ledger (provenance index — one row per hop·agent·chunk)
  ┌───────────┬────────────┬──────────┬───────────┬──────────┬──────────┬───────┐
  │ ledger_id │ session_id │ hop_num  │ from_agent│ to_agent │ chunk_id │ score │
  ├───────────┼────────────┼──────────┼───────────┼──────────┼──────────┼───────┤
  │  uuid-1   │  sess-001  │    0     │  router   │ billing  │c_order_42│ 0.91  │
  │  uuid-2   │  sess-001  │    0     │  router   │  fraud   │c_order_42│ 0.87  │
  │  uuid-3   │  sess-001  │    0     │  router   │  auth    │c_user_001│ 0.95  │
  └───────────┴────────────┴──────────┴───────────┴──────────┴──────────┴───────┘
        ▲ chunk_id referenced N times — content stored once above


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  SECURITY MODEL
  ──────────────

  Attack surface            Defense
  ─────────────             ───────
  WHERE clause injection    _safe_id() — regex [a-zA-Z0-9_\-.:]+  on ALL IDs
  Concurrent write race     threading.Lock (in-process) + FileLock (cross-process)
  LLM output poisoning      all fields clamped before manifest creation:
                              token_budget  → [100, 50000]
                              on_missing    → enum {fail_hard, warn, proceed}
                              history_depth → enum {full, decisions_only, ...}
                              must_carry    → list[:20], strings only
  JSON parse crash          try/except → heuristic fallback (no crash)
  PII leakage               exclude list applied at score time, hard -∞ penalty


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  INFERENCE PIPELINE (infer_manifest)
  ────────────────────────────────────

  Agent(role="Billing Specialist", goal="Resolve duplicate charges")
                     │
                     ▼
           ┌────────────────┐
           │  infer_manifest│
           └────────┬───────┘
                    │
           ┌────────▼────────────────────────────────────────┐
           │  llm_client provided?                           │
           │    YES → GPT-4o prompt → JSON parse             │
           │          (clamped fields, fallback on bad JSON) │
           │    NO  → heuristic keyword matching             │
           │          role="billing" → must_carry=[order_id] │
           │          role="fraud"   → must_carry=[fraud_score]│
           └─────────────────────────────────────────────────┘
                    │
                    ▼
           ContextManifest (validated, safe, within bounds)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  FRAMEWORK ADAPTERS
  ───────────────────

  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │    LangGraph     │  │     CrewAI       │  │     AutoGen      │
  │                  │  │                  │  │                  │
  │ @chp_node_       │  │ CHPCrewTask(     │  │ CHPConversable   │
  │  middleware(     │  │   fn,            │  │  Agent(          │
  │   manifest,      │  │   manifest,      │  │   name,          │
  │   ledger,        │  │   ledger,        │  │   manifest,      │
  │   embedder)      │  │   embedder)      │  │   ledger,        │
  │                  │  │                  │  │   embedder)      │
  └──────────────────┘  └──────────────────┘  └──────────────────┘
         All adapters: same manifest API · same ledger · same embedder


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  EMBEDDER OPTIONS
  ─────────────────

  StubEmbedder                    SentenceTransformerEmbedder
  ────────────                    ───────────────────────────
  zero vectors                    all-MiniLM-L6-v2 (default)
  zero dependencies               384-dim real semantic vectors
  exclude/must_carry work         full ANN semantic search works
  ~15-20% token reduction         40-70% token reduction
  tests / CI / no-GPU             production / GPU optional


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  LEDGER MAINTENANCE OPS
  ───────────────────────

  prune(session_id)              delete all ledger rows for closed session
  prune_older_than(iso_ts)       TTL — delete rows before timestamp
  prune_orphan_chunks()          O(n) bulk delete: unreferenced chunk content
  compact()                      LanceDB optimize() — merge delta files
  stats()                        {"ledger_rows": N, "chunk_rows": M}
  query_by_meaning(text, ...)    ANN vector search → top_k envelopes


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PROVEN RESULTS (12-agent, 7-hop pipeline)
  ──────────────────────────────────────────

  ✓  16-36% token reduction on specialist agents
  ✓  Zero PII leakage across all agents and hops
  ✓  Full audit trail recoverable from ledger at any point
  ✓  Content dedup: same chunk selected by N agents → stored once
  ✓  Concurrent fan-out: 50 threads, zero data corruption
  ✓  Injection prevention: 8 attack vectors blocked
  ✓  Framework-agnostic: LangGraph, CrewAI, AutoGen adapters

  Not yet benchmarked at scale:
  ○  1000 sessions/min throughput
  ○  10M ledger rows query latency
  ○  100K chunk embedding index rebuild time


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  RESEARCH CONTRIBUTION
  ──────────────────────

  Prior work: Agent A decides what to pass Agent B
              ↳ sender authority → leakage, waste, no auditability

  CHP:        Agent B declares its own requirements (RECEIVER AUTHORITY)
              ↳ first formal protocol for receiver-declared context selection
              ↳ Target: AAMAS 2027 (cs.MA primary)

  Key invariants:
    1. must_carry chunks always delivered if present in pool
    2. exclude chunks never delivered (hard block, score = -∞)
    3. every selected chunk has a full provenance record
    4. content deduplication is transparent to agents
```
