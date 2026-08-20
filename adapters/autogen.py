from __future__ import annotations
from chp.schema.context_manifest import ContextManifest
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope
from chp.engine.embedder import Embedder
from chp.engine.scorer import select_chunks, aselect_chunks, ScorerWeights
from chp.ledger.base import LedgerBackend


class CHPConversableAgent:
    def __init__(
        self,
        manifest: ContextManifest,
        ledger: LedgerBackend,
        embedder: Embedder,
        name: str = "chp-agent",
        weights: ScorerWeights | None = None,
    ) -> None:
        self.name = name
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
                selected_because=[f"autogen_hop:{hop}"],
                score=0.0, must_carry=False, token_cost=chunk.token_cost, ledger_id=None,
            )
            try:
                self._ledger.write(session_id, hop, chunk.source_agent, self._manifest.agent_id, envelope)
            except RuntimeError:
                pass

    def select_context(self, chunks: list[AnnotatedChunk], session_id: str, hop: int) -> list[AnnotatedChunk]:
        """Synchronous context selection — use in standard AutoGen GroupChat."""
        selected = select_chunks(
            chunks, self._manifest, self._embedder, self._weights,
            ledger=self._ledger, session_id=session_id,
        )
        self._write_envelopes(selected, session_id, hop)
        return selected

    async def aselect_context(self, chunks: list[AnnotatedChunk], session_id: str, hop: int) -> list[AnnotatedChunk]:
        """
        Async context selection for AutoGen async agents.

        Non-blocking: ledger_fallback retry uses asyncio.sleep,
        ledger writes offloaded to thread pool via awrite().
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
                selected_because=[f"autogen_async_hop:{hop}"],
                score=0.0, must_carry=False, token_cost=chunk.token_cost, ledger_id=None,
            )
            try:
                await self._ledger.awrite(session_id, hop, chunk.source_agent, self._manifest.agent_id, envelope)
            except RuntimeError:
                pass
        return selected
