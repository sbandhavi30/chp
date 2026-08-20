from __future__ import annotations
import asyncio
import logging
import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING
from chp.schema.context_manifest import ContextManifest
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.engine.embedder import Embedder
from chp.observability import CHPEvent, emit, Timer
from chp.pii import get_pii_filter, PIIFilter

if TYPE_CHECKING:
    from chp.ledger.base import LedgerBackend

log = logging.getLogger(__name__)


@dataclass
class ScorerWeights:
    alpha: float = 0.4    # semantic similarity
    beta: float = 0.2     # recency
    gamma: float = 0.3    # must_carry bonus
    delta: float = 0.05   # token cost penalty
    epsilon: float = 0.05  # exclude penalty


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def _recency_weight(chunks: list[AnnotatedChunk]) -> np.ndarray:
    turns = np.array([c.source_turn for c in chunks], dtype=np.float32)
    if turns.max() == turns.min():
        return np.ones(len(chunks), dtype=np.float32)
    return (turns - turns.min()) / (turns.max() - turns.min())


def score_chunks(
    chunks: list[AnnotatedChunk],
    manifest: ContextManifest,
    embedder: Embedder,
    weights: ScorerWeights | None = None,
) -> list[tuple[AnnotatedChunk, float]]:
    if not chunks:
        return []
    w = weights or ScorerWeights()
    req = manifest.requires

    chunk_texts = [c.content for c in chunks]
    tag_texts = req.domain_tags or ["general"]
    chunk_embs = embedder.embed(chunk_texts)
    tag_embs = embedder.embed(tag_texts)

    sim_matrix = _cosine_sim(chunk_embs, tag_embs)
    semantic_scores = sim_matrix.max(axis=1)
    recency = _recency_weight(chunks)

    scores = np.zeros(len(chunks), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        content_lower = chunk.content.lower()
        must_carry_hit = any(key.lower() in content_lower for key in req.must_carry)
        exclude_hit = any(ex.lower() in content_lower for ex in req.exclude)
        token_penalty = chunk.token_cost / max(manifest.token_budget, 1)

        scores[i] = (
            w.alpha * float(semantic_scores[i])
            + w.beta * float(recency[i])
            + w.gamma * float(must_carry_hit)
            - w.delta * token_penalty
            - w.epsilon * float(exclude_hit)
        )

    pairs = list(zip(chunks, scores.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def _accept_upstream(chunk: AnnotatedChunk, accept: bool | list[str]) -> bool:
    """Return True if this agent-output chunk passes the accept_upstream_output gate."""
    if accept is False:
        return False
    if accept is True:
        return True
    return chunk.source_agent in accept


def _envelope_to_chunk(envelope) -> AnnotatedChunk:
    """Convert a RationaleEnvelope recovered from ledger into an AnnotatedChunk."""
    return AnnotatedChunk(
        chunk_id=envelope.chunk_id,
        content=envelope.content,
        token_cost=envelope.token_cost,
        source_agent=envelope.source_agent,
        source_turn=envelope.source_turn,
        is_agent_output=True,   # recovered from ledger — treat as upstream output
    )


def _scan_session(
    key: str,
    session_id: str,
    ledger: "LedgerBackend",
    excluded_ids: set[str],
    exclude_terms: list[str],
) -> AnnotatedChunk | None:
    """Single ledger.query pass for one session. Returns first matching envelope or None."""
    try:
        envelopes = ledger.query(session_id)
    except Exception as exc:
        log.warning("CHP ledger_fallback: ledger.query failed for session=%s (%s)", session_id, exc)
        return None

    for envelope in envelopes:
        if envelope.chunk_id in excluded_ids:
            continue
        if any(ex.lower() in envelope.content.lower() for ex in exclude_terms):
            continue
        if key.lower() in envelope.content.lower():
            return _envelope_to_chunk(envelope)
    return None


def _ledger_fallback(
    key: str,
    session_id: str,
    ledger: "LedgerBackend",
    excluded_ids: set[str],
    exclude_terms: list[str],
    retry_attempts: int = 0,
    retry_delay_ms: int = 200,
    parent_session_id: str | None = None,
) -> AnnotatedChunk | None:
    """
    Three-case fallback for a missing must_carry key:

    Case A — same session, agent skipped: query current session immediately.
    Case B — parallel race: retry up to retry_attempts times with retry_delay_ms wait.
    Case C — cross-session: if still missing and parent_session_id given, query it.

    Exclude check fires on every candidate regardless of which session it came from.
    """
    import time

    # Case A + B: try current session, then retry if configured
    for attempt in range(retry_attempts + 1):
        chunk = _scan_session(key, session_id, ledger, excluded_ids, exclude_terms)
        if chunk is not None:
            log.info(
                "CHP ledger_fallback: recovered must_carry '%s' from session=%s "
                "(attempt=%d, chunk_id=%s, source_agent=%s)",
                key, session_id, attempt, chunk.chunk_id, chunk.source_agent,
            )
            return chunk
        if attempt < retry_attempts:
            log.info(
                "CHP ledger_fallback: must_carry '%s' not yet in session=%s -- "
                "retry %d/%d in %dms",
                key, session_id, attempt + 1, retry_attempts, retry_delay_ms,
            )
            time.sleep(retry_delay_ms / 1000.0)

    # Case C: exhausted retries on current session, try parent session
    if parent_session_id is not None:
        chunk = _scan_session(key, parent_session_id, ledger, excluded_ids, exclude_terms)
        if chunk is not None:
            log.info(
                "CHP ledger_fallback: recovered must_carry '%s' from parent_session=%s "
                "(chunk_id=%s, source_agent=%s)",
                key, parent_session_id, chunk.chunk_id, chunk.source_agent,
            )
            return chunk
        log.warning(
            "CHP ledger_fallback: must_carry '%s' not found in session=%s or parent_session=%s",
            key, session_id, parent_session_id,
        )
    else:
        log.warning(
            "CHP ledger_fallback: must_carry '%s' not found in session=%s after %d attempt(s) "
            "-- proceeding without it",
            key, session_id, retry_attempts + 1,
        )

    return None


def select_chunks(
    chunks: list[AnnotatedChunk],
    manifest: ContextManifest,
    embedder: Embedder,
    weights: ScorerWeights | None = None,
    ledger: "LedgerBackend | None" = None,
    session_id: str | None = None,
    pii_filter: PIIFilter | None = None,
) -> list[AnnotatedChunk]:
    req = manifest.requires

    emit(CHPEvent.SELECT_CHUNKS_CALLED, {
        "agent_id": manifest.agent_id,
        "session_id": session_id,
        "input_chunks": len(chunks),
        "token_budget": manifest.token_budget,
    })

    # Resolve PII filter: per-call arg > global registry > None
    _pii = pii_filter if pii_filter is not None else get_pii_filter()

    # Keyword exclude + semantic PII detection — fires on ALL chunks before routing.
    excluded_ids: set[str] = set()
    for c in chunks:
        keyword_hit = any(ex.lower() in c.content.lower() for ex in req.exclude)
        pii_hit = (_pii is not None and _pii.contains_pii(c.content))
        if keyword_hit or pii_hit:
            excluded_ids.add(c.chunk_id)
            emit(CHPEvent.CHUNK_EXCLUDED, {
                "agent_id": manifest.agent_id, "chunk_id": c.chunk_id,
                "session_id": session_id,
                "reason": "pii" if pii_hit else "keyword",
            })

    # ── Agent-output routing ──────────────────────────────────────────────────
    upstream_budget = max(1, int(manifest.token_budget * 0.20))
    upstream_chunks: list[AnnotatedChunk] = []
    upstream_used = 0
    for c in chunks:
        if not c.is_agent_output:
            continue
        if c.chunk_id in excluded_ids:
            continue
        if not _accept_upstream(c, req.accept_upstream_output):
            continue
        if upstream_used + c.token_cost <= upstream_budget:
            upstream_chunks.append(c)
            upstream_used += c.token_cost

    upstream_ids = {c.chunk_id for c in upstream_chunks}

    # ── Normal manifest-scored selection ─────────────────────────────────────
    pool = [c for c in chunks if not c.is_agent_output]

    recovered_chunks: list[AnnotatedChunk] = []
    must_carry_ids: set[str] = set()
    for key in req.must_carry:
        matched = [c for c in pool if key.lower() in c.content.lower() and c.chunk_id not in excluded_ids]
        if matched:
            must_carry_ids.add(matched[0].chunk_id)
            emit(CHPEvent.MUST_CARRY_DELIVERED, {
                "agent_id": manifest.agent_id, "key": key,
                "chunk_id": matched[0].chunk_id, "session_id": session_id,
            })
        elif manifest.on_missing == "fail_hard":
            raise ValueError(f"must_carry key '{key}' not found in any chunk")
        elif manifest.on_missing == "ledger_fallback":
            emit(CHPEvent.LEDGER_FALLBACK_TRIGGERED, {
                "agent_id": manifest.agent_id, "key": key, "session_id": session_id,
            })
            if ledger is not None and session_id is not None:
                recovered = _ledger_fallback(
                    key, session_id, ledger, excluded_ids, req.exclude,
                    retry_attempts=req.fallback_retry_attempts,
                    retry_delay_ms=req.fallback_retry_delay_ms,
                    parent_session_id=manifest.parent_session_id,
                )
                if recovered is not None:
                    recovered_chunks.append(recovered)
                    must_carry_ids.add(recovered.chunk_id)
                    pool = pool + [recovered]
                    emit(CHPEvent.LEDGER_FALLBACK_RECOVERED, {
                        "agent_id": manifest.agent_id, "key": key,
                        "chunk_id": recovered.chunk_id, "session_id": session_id,
                    })
                else:
                    emit(CHPEvent.MUST_CARRY_MISSED, {
                        "agent_id": manifest.agent_id, "key": key, "session_id": session_id,
                    })
            else:
                log.warning(
                    "CHP ledger_fallback: must_carry '%s' missing and no ledger provided "
                    "to select_chunks() (agent=%s) -- proceeding without it",
                    key, manifest.agent_id,
                )
                emit(CHPEvent.MUST_CARRY_MISSED, {
                    "agent_id": manifest.agent_id, "key": key, "session_id": session_id,
                })

    must_carry_chunks = [c for c in pool if c.chunk_id in must_carry_ids]
    remaining_budget = (
        manifest.token_budget
        - upstream_used
        - sum(c.token_cost for c in must_carry_chunks)
    )

    candidates = [
        c for c in pool
        if c.chunk_id not in must_carry_ids and c.chunk_id not in excluded_ids
    ]
    scored = score_chunks(candidates, manifest, embedder, weights)

    selected = list(upstream_chunks) + list(must_carry_chunks)
    for chunk, _ in scored:
        if remaining_budget <= 0:
            break
        if chunk.token_cost <= remaining_budget:
            selected.append(chunk)
            remaining_budget -= chunk.token_cost

    tokens_in  = sum(c.token_cost for c in chunks)
    tokens_out = sum(c.token_cost for c in selected)
    for chunk in selected:
        emit(CHPEvent.CHUNK_SELECTED, {
            "agent_id": manifest.agent_id, "chunk_id": chunk.chunk_id,
            "token_cost": chunk.token_cost, "session_id": session_id,
        })
    emit(CHPEvent.TOKEN_REDUCTION, {
        "agent_id": manifest.agent_id, "session_id": session_id,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "reduction_pct": round((1 - tokens_out / max(tokens_in, 1)) * 100, 1),
    })

    return selected


async def _async_ledger_fallback(
    key: str,
    session_id: str,
    ledger: "LedgerBackend",
    excluded_ids: set[str],
    exclude_terms: list[str],
    retry_attempts: int = 0,
    retry_delay_ms: int = 200,
    parent_session_id: str | None = None,
) -> AnnotatedChunk | None:
    """
    Async version of _ledger_fallback.
    Uses asyncio.sleep instead of time.sleep — never blocks the event loop.
    Delegates ledger I/O to asyncio.to_thread via ledger.aquery().
    """
    async def scan(sid: str) -> AnnotatedChunk | None:
        try:
            envelopes = await ledger.aquery(sid)
        except Exception as exc:
            log.warning("CHP async ledger_fallback: aquery failed for session=%s (%s)", sid, exc)
            return None
        for envelope in envelopes:
            if envelope.chunk_id in excluded_ids:
                continue
            if any(ex.lower() in envelope.content.lower() for ex in exclude_terms):
                continue
            if key.lower() in envelope.content.lower():
                return _envelope_to_chunk(envelope)
        return None

    for attempt in range(retry_attempts + 1):
        chunk = await scan(session_id)
        if chunk is not None:
            log.info(
                "CHP async ledger_fallback: recovered '%s' from session=%s (attempt=%d)",
                key, session_id, attempt,
            )
            return chunk
        if attempt < retry_attempts:
            log.info(
                "CHP async ledger_fallback: '%s' not yet in session=%s -- retry %d/%d in %dms",
                key, session_id, attempt + 1, retry_attempts, retry_delay_ms,
            )
            await asyncio.sleep(retry_delay_ms / 1000.0)   # non-blocking

    if parent_session_id is not None:
        chunk = await scan(parent_session_id)
        if chunk is not None:
            log.info(
                "CHP async ledger_fallback: recovered '%s' from parent_session=%s",
                key, parent_session_id,
            )
            return chunk
        log.warning(
            "CHP async ledger_fallback: '%s' not found in session=%s or parent_session=%s",
            key, session_id, parent_session_id,
        )
    else:
        log.warning(
            "CHP async ledger_fallback: '%s' not found in session=%s after %d attempt(s)",
            key, session_id, retry_attempts + 1,
        )
    return None


async def aselect_chunks(
    chunks: list[AnnotatedChunk],
    manifest: ContextManifest,
    embedder: Embedder,
    weights: ScorerWeights | None = None,
    ledger: "LedgerBackend | None" = None,
    session_id: str | None = None,
    pii_filter: PIIFilter | None = None,
) -> list[AnnotatedChunk]:
    """
    Async version of select_chunks.

    Identical logic to select_chunks but:
    - ledger_fallback uses asyncio.sleep (non-blocking retry)
    - ledger I/O via aquery() (offloaded to thread pool)
    - CPU-bound scoring runs in thread pool via asyncio.to_thread

    Drop-in replacement for async agent frameworks (LangGraph async nodes,
    FastAPI endpoints, async CrewAI).
    """
    req = manifest.requires

    emit(CHPEvent.SELECT_CHUNKS_CALLED, {
        "agent_id": manifest.agent_id,
        "session_id": session_id,
        "input_chunks": len(chunks),
        "token_budget": manifest.token_budget,
    })

    # Resolve PII filter: per-call arg > global registry > None
    _pii = pii_filter if pii_filter is not None else get_pii_filter()

    # Keyword exclude + semantic PII detection — fires on ALL chunks before routing.
    excluded_ids: set[str] = set()
    for c in chunks:
        keyword_hit = any(ex.lower() in c.content.lower() for ex in req.exclude)
        pii_hit = (_pii is not None and _pii.contains_pii(c.content))
        if keyword_hit or pii_hit:
            excluded_ids.add(c.chunk_id)
            emit(CHPEvent.CHUNK_EXCLUDED, {
                "agent_id": manifest.agent_id, "chunk_id": c.chunk_id,
                "session_id": session_id,
                "reason": "pii" if pii_hit else "keyword",
            })

    # ── Upstream output routing (same as sync) ────────────────────────────────
    upstream_budget = max(1, int(manifest.token_budget * 0.20))
    upstream_chunks: list[AnnotatedChunk] = []
    upstream_used = 0
    for c in chunks:
        if not c.is_agent_output:
            continue
        if c.chunk_id in excluded_ids:
            continue
        if not _accept_upstream(c, req.accept_upstream_output):
            continue
        if upstream_used + c.token_cost <= upstream_budget:
            upstream_chunks.append(c)
            upstream_used += c.token_cost

    pool = [c for c in chunks if not c.is_agent_output]

    # ── must_carry with async fallback ────────────────────────────────────────
    recovered_chunks: list[AnnotatedChunk] = []
    must_carry_ids: set[str] = set()
    for key in req.must_carry:
        matched = [c for c in pool if key.lower() in c.content.lower() and c.chunk_id not in excluded_ids]
        if matched:
            must_carry_ids.add(matched[0].chunk_id)
            emit(CHPEvent.MUST_CARRY_DELIVERED, {
                "agent_id": manifest.agent_id, "key": key,
                "chunk_id": matched[0].chunk_id, "session_id": session_id,
            })
        elif manifest.on_missing == "fail_hard":
            raise ValueError(f"must_carry key '{key}' not found in any chunk")
        elif manifest.on_missing == "ledger_fallback":
            emit(CHPEvent.LEDGER_FALLBACK_TRIGGERED, {
                "agent_id": manifest.agent_id, "key": key, "session_id": session_id,
            })
            if ledger is not None and session_id is not None:
                recovered = await _async_ledger_fallback(
                    key, session_id, ledger, excluded_ids, req.exclude,
                    retry_attempts=req.fallback_retry_attempts,
                    retry_delay_ms=req.fallback_retry_delay_ms,
                    parent_session_id=manifest.parent_session_id,
                )
                if recovered is not None:
                    recovered_chunks.append(recovered)
                    must_carry_ids.add(recovered.chunk_id)
                    pool = pool + [recovered]
                    emit(CHPEvent.LEDGER_FALLBACK_RECOVERED, {
                        "agent_id": manifest.agent_id, "key": key,
                        "chunk_id": recovered.chunk_id, "session_id": session_id,
                    })
                else:
                    emit(CHPEvent.MUST_CARRY_MISSED, {
                        "agent_id": manifest.agent_id, "key": key, "session_id": session_id,
                    })
            else:
                log.warning(
                    "CHP aselect_chunks: must_carry '%s' missing and no ledger provided "
                    "(agent=%s) -- proceeding without it",
                    key, manifest.agent_id,
                )
                emit(CHPEvent.MUST_CARRY_MISSED, {
                    "agent_id": manifest.agent_id, "key": key, "session_id": session_id,
                })

    # ── Scored selection (CPU-bound — offload to thread pool) ─────────────────
    must_carry_chunks = [c for c in pool if c.chunk_id in must_carry_ids]
    remaining_budget = (
        manifest.token_budget
        - upstream_used
        - sum(c.token_cost for c in must_carry_chunks)
    )
    candidates = [
        c for c in pool
        if c.chunk_id not in must_carry_ids and c.chunk_id not in excluded_ids
    ]
    scored = await asyncio.to_thread(score_chunks, candidates, manifest, embedder, weights)

    selected = list(upstream_chunks) + list(must_carry_chunks)
    for chunk, _ in scored:
        if remaining_budget <= 0:
            break
        if chunk.token_cost <= remaining_budget:
            selected.append(chunk)
            remaining_budget -= chunk.token_cost

    tokens_in  = sum(c.token_cost for c in chunks)
    tokens_out = sum(c.token_cost for c in selected)
    for chunk in selected:
        emit(CHPEvent.CHUNK_SELECTED, {
            "agent_id": manifest.agent_id, "chunk_id": chunk.chunk_id,
            "token_cost": chunk.token_cost, "session_id": session_id,
        })
    emit(CHPEvent.TOKEN_REDUCTION, {
        "agent_id": manifest.agent_id, "session_id": session_id,
        "tokens_in": tokens_in, "tokens_out": tokens_out,
        "reduction_pct": round((1 - tokens_out / max(tokens_in, 1)) * 100, 1),
    })

    return selected
