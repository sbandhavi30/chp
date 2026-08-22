# CHP — Context Handoff Protocol

**Manifest-driven context selection for multi-agent AI systems.**

Each agent declares what it needs. CHP selects, scores, and routes only the relevant context — without changing your agent code.

Token reduction depends on your pipeline shape. Run `python -m chp.benchmarks.compare` to measure your actual number.

---

## The Problem

Every production multi-agent system today uses **sender authority**: Agent A decides what to pass to Agent B. Agent B gets everything — or random chunks — regardless of what its task actually requires.

Result: context windows flood with irrelevant history, PII leaks across agent boundaries, costs balloon, and latency climbs.

## The Solution

CHP inverts control. **Agent B declares requirements** in a `ContextManifest` before context is ever selected. The CHP engine scores, filters, and routes accordingly.

```
Agent A output → [CHP SelectionEngine] → Agent B context
                        ↑
                ContextManifest (B's requirements)
                        ↑
                ContextLedger (provenance + fallback)
```

---

## Ledger Backend

CHP ships six ledger backends. Swap by passing any `LedgerBackend` to `CHPLedger(...)` — or just use the right class directly. The scorer, adapters, rate limiter, and lock backends all work with any backend.

| Backend | Install | Best for | Vector search |
|---------|---------|----------|---------------|
| `LanceDBLedger` (default) | `pip install chp` | Embedded, single/multi-node, ANN search | Yes (IVF_PQ) |
| `SQLiteLedger` | `pip install chp` | Single-node, zero extra deps, edge devices | No |
| `InMemoryLedger` | `pip install chp` | Tests, CI, ephemeral pipelines | No |
| `PostgresLedger` | `pip install "chp[postgres]"` | Teams already running Postgres, multi-node safe | No |
| `DynamoDBLedger` | `pip install "chp[dynamodb]"` | AWS-native deployments, serverless | No |
| `MongoDBLedger` | `pip install "chp[mongodb]"` | Teams already running MongoDB / Atlas | No |

```python
from chp import (
    LanceDBLedger, SQLiteLedger, InMemoryLedger,
    PostgresLedger, DynamoDBLedger, MongoDBLedger,
)

# LanceDB — default, embedded vector DB
ledger = LanceDBLedger("/data/chp")

# SQLite — zero extra deps
ledger = SQLiteLedger("/data/chp.db")

# InMemory — tests / CI
ledger = InMemoryLedger()

# Postgres — teams already running Postgres
ledger = PostgresLedger("postgresql://user:pass@host:5432/dbname")

# DynamoDB — AWS-native
ledger = DynamoDBLedger(table_prefix="chp", region_name="us-east-1")

# DynamoDB Local / LocalStack (dev/test)
ledger = DynamoDBLedger(
    table_prefix="chp",
    endpoint_url="http://localhost:8000",
    region_name="us-east-1",
)

# MongoDB
ledger = MongoDBLedger("mongodb://localhost:27017", db_name="chp")

# MongoDB Atlas
ledger = MongoDBLedger("mongodb+srv://user:pass@cluster.mongodb.net", db_name="chp")
```

**Swapping is one line** — everything else (scorer, adapters, rate limiter) stays unchanged:

```python
# Before
ledger = LanceDBLedger("/data/chp")

# After — switch to Postgres, nothing else changes
ledger = PostgresLedger("postgresql://user:pass@pg-svc:5432/mydb")
```

### Adding your own backend

Subclass `LedgerBackend` and implement 4 abstract methods:

```python
from chp.ledger.base import LedgerBackend

class MyRedisLedger(LedgerBackend):
    def write(self, session_id, hop_number, from_agent, to_agent, envelope, embedder=None): ...
    def query(self, session_id, agent_id=None): ...
    def query_hop(self, session_id, hop_number): ...
    def prune(self, session_id): ...
    # async awrite / aquery / etc. — inherited from base via asyncio.to_thread
```

---

## Deployment Topology

| Deployment | Lock backend | Ledger backend |
|------------|-------------|----------------|
| Single process (dev/test) | `NoOpLockBackend` | `InMemoryLedger` |
| Single node, multi-process | `FileLockBackend` (default) | `LanceDBLedger` or `SQLiteLedger` |
| Multi-node / Kubernetes / NFS / EFS | `RedisLockBackend` | `LanceDBLedger` + EFS, or `PostgresLedger` / `MongoDBLedger` |
| AWS serverless | n/a (DynamoDB handles it) | `DynamoDBLedger` |

```python
# Single-node (default — no config needed)
ledger = LanceDBLedger("/data/chp")

# Multi-node / Kubernetes with shared EFS
import redis
from chp import RedisLockBackend
r = redis.Redis.from_url("redis://redis-svc:6379/0")
ledger = LanceDBLedger("/mnt/efs/chp", lock_backend=RedisLockBackend(r))

# Multi-node — Postgres (no extra lock backend needed, Postgres handles concurrency)
ledger = PostgresLedger("postgresql://user:pass@pg-svc:5432/mydb")

# AWS serverless — DynamoDB (no extra lock backend needed)
ledger = DynamoDBLedger(table_prefix="prod-chp", region_name="us-east-1")

# Async high-throughput FastAPI service
from redis.asyncio import Redis
from chp import AsyncRedisLockBackend
r = Redis.from_url("redis://redis-svc:6379/0")
ledger = LanceDBLedger("/mnt/efs/chp", lock_backend=AsyncRedisLockBackend(r))
```

`RedisLockBackend` uses `SET NX PX` + Lua release script (standard Redlock lite). Safe on NFS, EFS, and any shared filesystem. Redis TTL auto-releases locks if a pod crashes mid-write.

---

## Install

```bash
pip install chp                          # core (LanceDB + SQLite + InMemory)
pip install "chp[embeddings]"            # + sentence-transformers
pip install "chp[llm]"                   # + openai (for infer_manifest)
pip install "chp[redis]"                 # + Redis lock backend
pip install "chp[postgres]"              # + PostgresLedger
pip install "chp[dynamodb]"              # + DynamoDBLedger
pip install "chp[mongodb]"               # + MongoDBLedger
pip install "chp[all]"                   # everything
```

Requires Python 3.11+.

---

## Quick Start

```python
from chp import ContextManifest, ContextRequirements, AnnotatedChunk, infer_manifest
from chp.engine.scorer import select_chunks
from chp.engine.embedder import StubEmbedder

# Agent B declares what it needs
manifest = ContextManifest(
    agent_id="billing-agent",
    task="Process payment and generate invoice",
    requires=ContextRequirements(
        must_carry=["billing_decision", "customer_id"],
        domain_tags=["billing", "payment"],
        exclude=["SSN", "password"],
        accept_upstream_output=["auth-agent"],   # accept auth agent's output
    ),
    token_budget=2000,
    on_missing="ledger_fallback",
)

# Context pool from upstream
chunks = [
    AnnotatedChunk(chunk_id="c1", content="customer_id: 42 billing_decision: approved",
                   token_cost=50, source_agent="auth-agent", source_turn=1),
    AnnotatedChunk(chunk_id="c2", content="irrelevant system log entry",
                   token_cost=200, source_agent="router", source_turn=0),
    AnnotatedChunk(chunk_id="c3", content="payment amount: $150",
                   token_cost=30, source_agent="router", source_turn=2),
]

selected = select_chunks(chunks, manifest, StubEmbedder())
# → c1 (must_carry), c3 (semantic match) — c2 filtered out
```

### Auto-infer a manifest from role description

```python
manifest = infer_manifest(
    agent_id="fraud-detector",
    role="Fraud detection agent for financial transactions",
    goal="Flag suspicious transactions before payment processing",
    token_budget=3000,
    pipeline_agents=["auth-agent", "billing-agent", "fraud-detector"],
)
# → manifest with domain_tags, must_carry keys, accept_upstream_output=["billing-agent"]
```

---

## Framework Adapters

### LangGraph

```python
from chp.adapters.langgraph import achp_node_middleware
from chp.ledger.lancedb_ledger import CHPLedger

ledger = CHPLedger()

@achp_node_middleware(manifest=billing_manifest, ledger=ledger, embedder=embedder)
async def billing_node(state: dict) -> dict:
    chunks = state["chp_selected_chunks"]   # CHP-filtered context
    result = await llm.ainvoke(build_prompt(chunks))
    return {**state, "result": result}

# State keys CHP reads/writes:
#   state["chp_chunks"]           = list[AnnotatedChunk]   (input pool)
#   state["chp_session_id"]       = str
#   state["chp_hop"]              = int  (auto-incremented)
#   state["chp_selected_chunks"]  = list[AnnotatedChunk]   (CHP output)
```

Sync version: `chp_node_middleware` (same signature, no `async`).

### CrewAI

```python
from chp.adapters.crewai import CHPCrewTask

task = CHPCrewTask(
    task_fn=my_crew_task,
    manifest=manifest,
    ledger=ledger,
    embedder=embedder,
)

result = task.run(chunks, session_id="session-1", hop=0)       # sync
result = await task.arun(chunks, session_id="session-1", hop=0) # async
```

### AutoGen

```python
from chp.adapters.autogen import CHPConversableAgent

agent = CHPConversableAgent(
    manifest=manifest, ledger=ledger, embedder=embedder,
    name="billing-conversable",
)

selected = agent.select_context(chunks, session_id="s1", hop=0)        # sync
selected = await agent.aselect_context(chunks, session_id="s1", hop=0) # async
```

---

## Ledger Fallback

When a `must_carry` key is missing from the current context pool, CHP can recover it from the ledger instead of failing.

```python
manifest = ContextManifest(
    agent_id="fraud-agent",
    requires=ContextRequirements(
        must_carry=["billing_decision"],
        fallback_retry_attempts=3,     # retry up to 3× (for parallel race conditions)
        fallback_retry_delay_ms=200,   # wait 200ms between retries
    ),
    on_missing="ledger_fallback",
    parent_session_id="parent-session-xyz",  # Case C: cross-session recovery
)
```

Three recovery cases:

| Case | Scenario | Behavior |
|------|----------|----------|
| A | Agent skipped in same session | Immediate ledger query |
| B | Parallel race — writer not yet committed | Retry loop with configurable delay |
| C | Cross-session dependency | Query `parent_session_id` after retries exhausted |

---

## Upstream Agent Output

By default agents don't see each other's generated outputs. Opt in per-agent:

```python
# Accept output from any upstream agent (capped at 20% of token_budget)
requires = ContextRequirements(accept_upstream_output=True)

# Accept only specific agents
requires = ContextRequirements(accept_upstream_output=["auth-agent", "billing-agent"])

# Block all upstream output (default)
requires = ContextRequirements(accept_upstream_output=False)
```

Mark a chunk as agent-generated output:

```python
chunk = AnnotatedChunk(..., is_agent_output=True)
```

Exclude rules fire on upstream output too — PII never leaks regardless of `accept_upstream_output`.

---

## PII Detection

CHP ships two PII filters that intercept chunks **before** routing, scoring, or ledger writes. Unlike the `exclude` keyword list, these detect semantically equivalent PII that a keyword scan would miss (e.g. `exclude=["SSN"]` misses `"Social Security: 123-45-6789"`).

### RegexPIIFilter — zero dependencies

```python
from chp.pii import RegexPIIFilter, set_pii_filter

# Global hook — applies to every select_chunks() call
set_pii_filter(RegexPIIFilter())

# Scope to specific entity types
set_pii_filter(RegexPIIFilter(enabled_types=["ssn", "credit_card", "email"]))

# Per-call override (bypasses the global filter)
from chp.engine.scorer import select_chunks
selected = select_chunks(chunks, manifest, embedder, pii_filter=my_filter)

# Deregister
set_pii_filter(None)
```

Covers 15 entity types:

| Type | Examples detected |
|------|------------------|
| `ssn` | `123-45-6789`, `123456789`, "Social Security" |
| `credit_card` | Visa, Mastercard, Amex, Discover |
| `email` | `user@example.com` |
| `phone_us` | `(555) 123-4567`, `555-123-4567` |
| `phone_intl` | `+44 7700 900000` |
| `ipv4` | `192.168.1.100` |
| `ipv6` | `2001:db8::1` |
| `dob` | "Date of Birth:", "born on", "DOB:" |
| `passport` | `A1234567`, `AB123456` |
| `drivers_license` | "driver's license ABC-12345" |
| `bank_account` | "account number 123456789012", IBAN |
| `routing_number` | "routing number 021000021" |
| `mrn` | "MRN: ABC-12345" |
| `api_key` | `api_key: sk-abc...`, bearer tokens |
| `password` | `password: hunter2` |

### PresidioPIIFilter — ML-powered (50+ entity types)

```python
# pip install "chp[pii]"
# python -m spacy download en_core_web_lg

from chp.pii import PresidioPIIFilter, set_pii_filter

set_pii_filter(PresidioPIIFilter())

# Scope to specific Presidio entity types
set_pii_filter(PresidioPIIFilter(
    entities=["US_SSN", "CREDIT_CARD", "EMAIL_ADDRESS", "PERSON"],
    score_threshold=0.7,
))
```

### Debugging detected types

```python
f = RegexPIIFilter()
f.detected_types("contact user@example.com, SSN 123-45-6789")
# → ["EMAIL", "SSN"]
```

### CHUNK_EXCLUDED event — PII vs keyword reason

The `CHUNK_EXCLUDED` metrics event now carries a `reason` field:

```python
def my_hook(event, data):
    if event == chp.CHPEvent.CHUNK_EXCLUDED:
        print(data["reason"])   # "pii" or "keyword"
        print(data["chunk_id"])

chp.set_metrics_hook(my_hook)
```

---

## Metrics Hook

Zero-dependency integration with Prometheus, Datadog, StatsD, or any metrics backend:

```python
import chp
from prometheus_client import Counter, Histogram

tokens_saved = Counter("chp_tokens_saved_total", "Tokens filtered by CHP", ["agent_id"])
write_latency = Histogram("chp_ledger_write_seconds", "Ledger write latency", ["agent_id"])

def my_hook(event: str, data: dict) -> None:
    if event == chp.CHPEvent.TOKEN_REDUCTION:
        saved = data["tokens_in"] - data["tokens_out"]
        tokens_saved.labels(agent_id=data["agent_id"]).inc(saved)
    elif event == chp.CHPEvent.LEDGER_WRITE:
        write_latency.labels(agent_id=data.get("to_agent", "")).observe(data["elapsed_ms"] / 1000)

chp.set_metrics_hook(my_hook)
```

### Available events

| Event constant | When fired | Key data fields |
|----------------|-----------|-----------------|
| `CHPEvent.SELECT_CHUNKS_CALLED` | Entry to select_chunks | `agent_id`, `input_chunks`, `token_budget` |
| `CHPEvent.CHUNK_SELECTED` | Each chunk included | `agent_id`, `chunk_id`, `token_cost` |
| `CHPEvent.CHUNK_EXCLUDED` | Each chunk filtered (exclude list or PII) | `agent_id`, `chunk_id`, `reason` ("keyword"\|"pii") |
| `CHPEvent.MUST_CARRY_DELIVERED` | must_carry key found in pool | `agent_id`, `key`, `chunk_id` |
| `CHPEvent.MUST_CARRY_MISSED` | must_carry key not found anywhere | `agent_id`, `key` |
| `CHPEvent.LEDGER_FALLBACK_TRIGGERED` | Fallback search started | `agent_id`, `key`, `session_id` |
| `CHPEvent.LEDGER_FALLBACK_RECOVERED` | Fallback found the chunk | `agent_id`, `key`, `chunk_id` |
| `CHPEvent.LEDGER_WRITE` | Ledger row written | `session_id`, `hop_number`, `from_agent`, `to_agent`, `elapsed_ms` |
| `CHPEvent.LEDGER_QUERY` | Ledger queried | `session_id`, `rows_returned`, `elapsed_ms` |
| `CHPEvent.TOKEN_REDUCTION` | End of select_chunks | `agent_id`, `tokens_in`, `tokens_out`, `reduction_pct` |

All events also emit a structured JSON log line via Python's standard `logging` module (logger `chp.observability`).

---

## Per-Session Token Reporting

`SessionTokenTracker` accumulates `TOKEN_REDUCTION` events across all hops in a session and emits a final summary — giving users "X% on your pipeline" rather than a benchmark claim.

```python
import chp
from chp import SessionTokenTracker

tracker = SessionTokenTracker("session-abc")
chp.set_metrics_hook(tracker.on_event)

# ... run your pipeline (each select_chunks call fires TOKEN_REDUCTION) ...

summary = tracker.close()
# {
#   "session_id": "session-abc",
#   "hops": 5,
#   "total_tokens_in": 4775,
#   "total_tokens_out": 1500,
#   "overall_reduction_pct": 68.6
# }
```

Chain with an existing Prometheus/Datadog hook — tracker forwards all events upstream:

```python
tracker = SessionTokenTracker("s1", upstream_hook=my_prometheus_hook)
chp.set_metrics_hook(tracker.on_event)
```

Reset between sessions:

```python
tracker.reset()  # reuse for next session
```

---

## Scorer Weights

```python
from chp.engine.scorer import ScorerWeights

weights = ScorerWeights(
    alpha=0.4,    # semantic similarity (cosine vs domain_tags)
    beta=0.2,     # recency (higher source_turn = more recent)
    gamma=0.3,    # must_carry bonus
    delta=0.05,   # token cost penalty
    epsilon=0.05, # exclude penalty
)

selected = select_chunks(chunks, manifest, embedder, weights=weights)
```

---

## Ledger

```python
from chp.ledger.lancedb_ledger import CHPLedger
from chp.engine.embedder import SentenceTransformerEmbedder

ledger = CHPLedger(db_path="/data/chp_ledger")  # persistent
ledger = CHPLedger()                             # temp dir (testing)

# Write
ledger_id = ledger.write(session_id, hop, from_agent, to_agent, envelope, embedder=embedder)

# Query by session
envelopes = ledger.query(session_id)
envelopes = ledger.query(session_id, agent_id="billing-agent")

# Semantic search
envelopes = ledger.query_by_meaning("payment declined", embedder, session_id=session_id, top_k=5)

# Maintenance
ledger.prune(session_id)            # delete session rows
ledger.prune_orphan_chunks()        # reclaim unreferenced chunk content
ledger.compact()                    # merge delta files, rebuild vector index

# Async API (all sync methods have async counterparts)
await ledger.awrite(...)
await ledger.aquery(session_id)
await ledger.aquery_by_meaning(...)
await ledger.aprune(session_id)
```

Two-table LanceDB design:
- `chp_chunks` — content + embedding, deduped by `chunk_id`
- `chp_ledger` — provenance index (session, hop, agent pair)

---

## Embedders

```python
from chp.engine.embedder import StubEmbedder, SentenceTransformerEmbedder

# Testing / no dependencies
embedder = StubEmbedder()

# Production — pip install "chp[embeddings]"
embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
```

---

## Architecture

See [`docs/chp-architecture.md`](../docs/chp-architecture.md) for the full ASCII diagram covering all layers, topologies, LanceDB internals, and security model.

---

## Running Tests

```bash
# Unit tests (no API keys needed)
.venv/bin/pytest chp/tests/unit/ -q

# Load tests (100 sessions synthetic)
.venv/bin/pytest chp/tests/load/test_load_100sessions.py -v -s

# Load tests with real LLM (needs OPENAI_API_KEY)
OPENAI_API_KEY=sk-... .venv/bin/pytest chp/tests/load/test_load_llm.py -v -s
```

---

## License

Apache 2.0
