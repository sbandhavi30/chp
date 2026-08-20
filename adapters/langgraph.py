from __future__ import annotations
from functools import wraps
from typing import Any, Callable
from chp.schema.context_manifest import ContextManifest
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope
from chp.engine.embedder import Embedder
from chp.engine.scorer import select_chunks, aselect_chunks, ScorerWeights
from chp.ledger.base import LedgerBackend


def _build_envelopes(
    selected: list[AnnotatedChunk],
    chunks: list[AnnotatedChunk],
    manifest: ContextManifest,
    hop: int,
) -> list[RationaleEnvelope]:
    must_carry_ids = {
        c.chunk_id for c in chunks
        if any(k.lower() in c.content.lower() for k in manifest.requires.must_carry)
    }
    return [
        RationaleEnvelope(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            source_agent=chunk.source_agent,
            source_turn=chunk.source_turn,
            hop_sequence=[chunk.source_agent, manifest.agent_id],
            selected_because=[f"chp_selected_hop:{hop}"],
            score=0.0,
            must_carry=chunk.chunk_id in must_carry_ids,
            token_cost=chunk.token_cost,
            ledger_id=None,
        )
        for chunk in selected
    ]


def chp_node_middleware(
    manifest: ContextManifest,
    ledger: LedgerBackend,
    embedder: Embedder,
    weights: ScorerWeights | None = None,
) -> Callable:
    """Sync decorator for synchronous LangGraph nodes."""
    def decorator(node_fn: Callable) -> Callable:
        @wraps(node_fn)
        def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            chunks: list[AnnotatedChunk] = state.get("chp_chunks", [])
            session_id: str = state.get("chp_session_id", "default")
            hop: int = state.get("chp_hop", 0)

            selected = select_chunks(
                chunks, manifest, embedder, weights,
                ledger=ledger, session_id=session_id,
            )
            for envelope in _build_envelopes(selected, chunks, manifest, hop):
                try:
                    ledger.write(session_id, hop, envelope.source_agent, manifest.agent_id, envelope)
                except RuntimeError:
                    pass

            state["chp_selected_chunks"] = selected
            state["chp_hop"] = hop + 1
            return node_fn(state)
        return wrapper
    return decorator


def achp_node_middleware(
    manifest: ContextManifest,
    ledger: LedgerBackend,
    embedder: Embedder,
    weights: ScorerWeights | None = None,
) -> Callable:
    """
    Async decorator for async LangGraph nodes (production pattern).

    Usage:
        @achp_node_middleware(manifest=billing_manifest, ledger=ledger, embedder=embedder)
        async def billing_node(state: dict) -> dict:
            chunks = state["chp_selected_chunks"]
            result = await llm.ainvoke(...)
            return state

    State keys:
        state["chp_chunks"]      = list[AnnotatedChunk]
        state["chp_session_id"]  = str
        state["chp_hop"]         = int  (auto-incremented)
    """
    def decorator(node_fn: Callable) -> Callable:
        @wraps(node_fn)
        async def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            chunks: list[AnnotatedChunk] = state.get("chp_chunks", [])
            session_id: str = state.get("chp_session_id", "default")
            hop: int = state.get("chp_hop", 0)

            selected = await aselect_chunks(
                chunks, manifest, embedder, weights,
                ledger=ledger, session_id=session_id,
            )
            for envelope in _build_envelopes(selected, chunks, manifest, hop):
                try:
                    await ledger.awrite(session_id, hop, envelope.source_agent, manifest.agent_id, envelope)
                except RuntimeError:
                    pass

            state["chp_selected_chunks"] = selected
            state["chp_hop"] = hop + 1
            return await node_fn(state)
        return wrapper
    return decorator
