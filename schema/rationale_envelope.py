from __future__ import annotations
from datetime import datetime, timezone
from pydantic import BaseModel, Field


_AGENT_OUTPUT_TAG = "__agent_output__"


class AnnotatedChunk(BaseModel):
    chunk_id: str
    content: str
    token_cost: int
    source_agent: str
    source_turn: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # Set True when this chunk IS the output produced by source_agent (not ambient context).
    # Routed via accept_upstream_output logic instead of normal manifest scoring.
    is_agent_output: bool = False


class RationaleEnvelope(BaseModel):
    chunk_id: str
    content: str
    source_agent: str
    source_turn: int
    hop_sequence: list[str]
    selected_because: list[str]
    score: float = Field(ge=0.0, le=1.0)
    must_carry: bool
    token_cost: int
    ledger_id: str | None = None
