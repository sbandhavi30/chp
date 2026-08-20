from __future__ import annotations
from typing import Callable, Any
from chp.schema.context_manifest import ContextManifest
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope
from chp.engine.embedder import Embedder
from chp.engine.scorer import select_chunks, aselect_chunks, ScorerWeights
from chp.ledger.base import LedgerBackend


class CHPCrewTask:
    def __init__(
        self,
        task_fn: Callable,
        manifest: ContextManifest,
        ledger: LedgerBackend,
        embedder: Embedder,
        weights: ScorerWeights | None = None,
    ) -> None:
        self._task_fn = task_fn
        self._manifest = manifest
        self._ledger = ledger
        self._embedder = embedder
        self._weights = weights

    def _write_envelopes(self, selected: list[AnnotatedChunk], session_id: str, hop: int) -> None:
        for chunk in selected:
            envelope = RationaleEnvelope(
                chunk_id=chunk.chunk_id, content=chunk.content,
                source_agent=chunk.source_agent, source_turn=chunk.source_turn,
                hop_sequence=[chunk.source_agent, self._manifest.agent_id],
                selected_because=[f"crewai_hop:{hop}"],
                score=0.0, must_carry=False, token_cost=chunk.token_cost, ledger_id=None,
            )
            try:
                self._ledger.write(session_id, hop, chunk.source_agent, self._manifest.agent_id, envelope)
            except RuntimeError:
                pass

    def run(self, chunks: list[AnnotatedChunk], session_id: str, hop: int) -> Any:
        """Synchronous task execution."""
        selected = select_chunks(
            chunks, self._manifest, self._embedder, self._weights,
            ledger=self._ledger, session_id=session_id,
        )
        self._write_envelopes(selected, session_id, hop)
        return self._task_fn(selected)

    async def arun(self, chunks: list[AnnotatedChunk], session_id: str, hop: int) -> Any:
        """
        Async task execution for async CrewAI crews.

        Uses aselect_chunks() — non-blocking ledger_fallback retry,
        ledger writes offloaded to thread pool.
        """
        selected = await aselect_chunks(
            chunks, self._manifest, self._embedder, self._weights,
            ledger=self._ledger, session_id=session_id,
        )
        for chunk in selected:
            envelope = RationaleEnvelope(
                chunk_id=chunk.chunk_id, content=chunk.content,
                source_agent=chunk.source_agent, source_turn=chunk.source_turn,
                hop_sequence=[chunk.source_agent, self._manifest.agent_id],
                selected_because=[f"crewai_async_hop:{hop}"],
                score=0.0, must_carry=False, token_cost=chunk.token_cost, ledger_id=None,
            )
            try:
                await self._ledger.awrite(session_id, hop, chunk.source_agent, self._manifest.agent_id, envelope)
            except RuntimeError:
                pass
        # task_fn may be sync or async
        import asyncio, inspect
        if inspect.iscoroutinefunction(self._task_fn):
            return await self._task_fn(selected)
        return await asyncio.to_thread(self._task_fn, selected)
