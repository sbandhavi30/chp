"""
CHP vs Baseline benchmark — synthetic 4-agent pipeline.

Measures token reduction, must_carry recall, PII containment, and scorer
latency on a reproducible chunk pool with known ground truth.

Run:
    python -m chp.benchmarks.compare
    python -m chp.benchmarks.compare --json results.json
    python -m chp.benchmarks.compare --hops 10

No external dependencies — uses StubEmbedder + InMemoryLedger.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import Callable

from chp.schema.rationale_envelope import AnnotatedChunk
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.engine.embedder import StubEmbedder
from chp.engine.scorer import select_chunks
from chp.ledger.memory_ledger import InMemoryLedger
from chp.pii import RegexPIIFilter


# ── Synthetic pipeline definition ─────────────────────────────────────────────
#
# 4-agent pipeline: Router → Auth → Billing → Summarizer
# Each hop has a known "relevant" set and known PII chunks.
# Baseline = pass everything. CHP = manifest-declared selection.

_AGENTS = ["router", "auth-agent", "billing-agent", "summarizer"]


def _make_chunk_pool(hop: int) -> list[AnnotatedChunk]:
    """Build a realistic chunk pool for one hop. ~20 chunks, 3 relevant, 2 PII."""
    base_turn = hop * 3
    return [
        # ── Relevant chunks ──────────────────────────────────────────────────
        AnnotatedChunk(
            chunk_id=f"h{hop}-customer",
            content=f"customer_id: CUST-{1000+hop} status: active tier: premium",
            token_cost=45, source_agent="auth-agent", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-billing",
            content=f"billing_decision: approved amount: ${150+hop*10} invoice_id: INV-{hop}",
            token_cost=60, source_agent="billing-agent", source_turn=base_turn+1,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-payment",
            content=f"payment_method: card_ending_4242 currency: USD hop={hop}",
            token_cost=40, source_agent="billing-agent", source_turn=base_turn+1,
        ),
        # ── PII chunks (should never pass through) ───────────────────────────
        AnnotatedChunk(
            chunk_id=f"h{hop}-ssn",
            content=f"Social Security Number: {800+hop:03d}-45-6789 customer_id: CUST-{1000+hop}",
            token_cost=35, source_agent="auth-agent", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-email",
            content=f"customer email: user{hop}@example.com password: s3cr3t{hop}",
            token_cost=30, source_agent="router", source_turn=base_turn,
        ),
        # ── Noise chunks (irrelevant) ─────────────────────────────────────────
        AnnotatedChunk(
            chunk_id=f"h{hop}-syslog1",
            content=f"[INFO] hop={hop} scheduler heartbeat tick=12345 worker=w{hop}",
            token_cost=55, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-syslog2",
            content=f"[DEBUG] memory_used=1.2GB cpu=43% pid={hop*100} queue_depth=0",
            token_cost=60, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-syslog3",
            content=f"[INFO] agent router initialized config_version=2.1 hop={hop}",
            token_cost=50, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-trace1",
            content=f"trace_id=abc{hop} span=http.request latency_ms=12 status=200",
            token_cost=45, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-trace2",
            content=f"trace_id=abc{hop} span=db.query table=sessions rows=1 latency_ms=3",
            token_cost=40, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-infra1",
            content=f"k8s pod billing-{hop} node=node-3 namespace=prod restarts=0",
            token_cost=55, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-infra2",
            content=f"k8s pod auth-{hop} node=node-1 namespace=prod cpu_request=250m",
            token_cost=50, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-metrics1",
            content=f"chp_tokens_saved_total{{agent=router}} {hop*120} hop={hop}",
            token_cost=35, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-metrics2",
            content=f"http_requests_total{{method=POST,status=200}} {hop*50}",
            token_cost=40, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-config1",
            content=f"feature_flag: new_checkout=false rollout_pct=0 env=prod",
            token_cost=45, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-config2",
            content=f"rate_limit: requests_per_minute=1000 burst=200 agent=billing",
            token_cost=40, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-history1",
            content=f"prior_session: session-{hop-1} outcome=success agent=auth duration_ms=230",
            token_cost=55, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-history2",
            content=f"prior_session: session-{hop-2} outcome=timeout agent=summarizer retry=1",
            token_cost=50, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-unused1",
            content=f"product_catalog: item_id=SKU-{hop} category=electronics price=299",
            token_cost=60, source_agent="router", source_turn=base_turn,
        ),
        AnnotatedChunk(
            chunk_id=f"h{hop}-unused2",
            content=f"recommendation_engine: user_id=USR-{hop} top_items=[SKU-1,SKU-2]",
            token_cost=65, source_agent="router", source_turn=base_turn,
        ),
    ]


def _billing_manifest() -> ContextManifest:
    return ContextManifest(
        agent_id="billing-agent",
        task="Process payment and generate invoice",
        requires=ContextRequirements(
            must_carry=["billing_decision", "customer_id"],
            domain_tags=["billing", "payment", "customer"],
            exclude=["SSN", "password"],
        ),
        token_budget=300,
    )


_PII_IDS = {"ssn", "email"}  # chunk_id suffix patterns that are PII


# ── Measurement helpers ───────────────────────────────────────────────────────

@dataclass
class HopResult:
    hop: int
    baseline_tokens: int
    chp_tokens: int
    reduction_pct: float
    must_carry_recall: float   # fraction of must_carry keys found
    pii_leaked: bool           # any PII chunk made it through
    scorer_latency_ms: float


@dataclass
class PipelineResult:
    hops: int
    total_baseline_tokens: int
    total_chp_tokens: int
    overall_reduction_pct: float
    avg_must_carry_recall: float
    pii_leaked_any_hop: bool
    avg_scorer_latency_ms: float
    per_hop: list[HopResult]


def _run_pipeline(num_hops: int) -> PipelineResult:
    manifest = _billing_manifest()
    embedder = StubEmbedder()
    pii_filter = RegexPIIFilter(log_detections=False)
    hop_results: list[HopResult] = []

    for hop in range(num_hops):
        pool = _make_chunk_pool(hop)
        baseline_tokens = sum(c.token_cost for c in pool)

        t0 = time.perf_counter()
        selected = select_chunks(pool, manifest, embedder, pii_filter=pii_filter)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        chp_tokens = sum(c.token_cost for c in selected)
        reduction = (baseline_tokens - chp_tokens) / baseline_tokens * 100

        # must_carry recall — did required keys land in selected?
        mc_keys = manifest.requires.must_carry
        found = sum(
            1 for key in mc_keys
            if any(key.lower() in c.content.lower() for c in selected)
        )
        recall = found / max(len(mc_keys), 1)

        # PII check — did any known-PII chunk survive selection?
        pii_leaked = any(
            any(pii_id in c.chunk_id for pii_id in _PII_IDS)
            for c in selected
        )

        hop_results.append(HopResult(
            hop=hop,
            baseline_tokens=baseline_tokens,
            chp_tokens=chp_tokens,
            reduction_pct=round(reduction, 1),
            must_carry_recall=round(recall, 3),
            pii_leaked=pii_leaked,
            scorer_latency_ms=round(elapsed_ms, 2),
        ))

    total_baseline = sum(h.baseline_tokens for h in hop_results)
    total_chp = sum(h.chp_tokens for h in hop_results)
    overall_reduction = (total_baseline - total_chp) / total_baseline * 100

    return PipelineResult(
        hops=num_hops,
        total_baseline_tokens=total_baseline,
        total_chp_tokens=total_chp,
        overall_reduction_pct=round(overall_reduction, 1),
        avg_must_carry_recall=round(
            sum(h.must_carry_recall for h in hop_results) / num_hops, 3
        ),
        pii_leaked_any_hop=any(h.pii_leaked for h in hop_results),
        avg_scorer_latency_ms=round(
            sum(h.scorer_latency_ms for h in hop_results) / num_hops, 2
        ),
        per_hop=hop_results,
    )


# ── Output formatting ─────────────────────────────────────────────────────────

def _print_results(r: PipelineResult) -> None:
    print()
    print("=" * 70)
    print("  CHP vs Baseline — Context Handoff Protocol Benchmark")
    print("=" * 70)
    print(f"  Pipeline:   4-agent synthetic (Router→Auth→Billing→Summarizer)")
    print(f"  Hops:       {r.hops}")
    print(f"  Embedder:   StubEmbedder (zero-dep, reproducible)")
    print(f"  Chunks/hop: 20  (3 relevant, 2 PII, 15 noise)")
    print()
    print(f"  {'Metric':<32} {'Baseline':>12} {'CHP':>12}")
    print(f"  {'-'*32} {'-'*12} {'-'*12}")
    print(f"  {'Total tokens':<32} {r.total_baseline_tokens:>12,} {r.total_chp_tokens:>12,}")
    print(f"  {'Token reduction':<32} {'—':>12} {r.overall_reduction_pct:>11.1f}%")
    print(f"  {'must_carry recall':<32} {'100%':>12} {r.avg_must_carry_recall*100:>11.1f}%")
    print(f"  {'PII leaked':<32} {'YES':>12} {'NO' if not r.pii_leaked_any_hop else 'YES':>12}")
    print(f"  {'Avg scorer latency':<32} {'0 ms':>12} {r.avg_scorer_latency_ms:>11.2f} ms")
    print()
    print(f"  Per-hop breakdown:")
    print(f"  {'Hop':<6} {'Baseline':>10} {'CHP':>8} {'Reduction':>10} {'Recall':>8} {'PII':>6} {'ms':>8}")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*6} {'-'*8}")
    for h in r.per_hop:
        pii = "LEAK" if h.pii_leaked else "OK"
        print(
            f"  {h.hop:<6} {h.baseline_tokens:>10,} {h.chp_tokens:>8,} "
            f"{h.reduction_pct:>9.1f}% {h.must_carry_recall*100:>7.1f}% "
            f"{pii:>6} {h.scorer_latency_ms:>7.2f}"
        )
    print()
    print("  Note: reduction varies by pipeline shape, chunk pool composition,")
    print("  and manifest specificity. Run on your own pipeline for real numbers.")
    print("=" * 70)
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark CHP vs baseline context passing"
    )
    parser.add_argument("--hops", type=int, default=5, help="Number of pipeline hops (default 5)")
    parser.add_argument("--json", metavar="FILE", help="Write results to JSON file")
    args = parser.parse_args()

    result = _run_pipeline(args.hops)
    _print_results(result)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(asdict(result), f, indent=2)
        print(f"  Results written to {args.json}")
        print()


if __name__ == "__main__":
    main()
