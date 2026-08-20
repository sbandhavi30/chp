from __future__ import annotations
from typing import Literal, Union
from pydantic import BaseModel, Field


class ContextRequirements(BaseModel):
    must_carry: list[str] = []
    domain_tags: list[str] = []
    history_depth: Literal["full", "decisions_only", "summary", "none"] = "full"
    recency_window: str | None = None
    exclude: list[str] = []
    # False  → ignore all upstream agent output (default, backward-compatible)
    # True   → accept output from any upstream agent
    # [...]  → accept only from named agent IDs
    accept_upstream_output: Union[bool, list[str]] = False
    # Case B: parallel race — how many times to retry ledger query before giving up.
    # Each retry waits fallback_retry_delay_ms milliseconds (default 200ms).
    # Only active when on_missing="ledger_fallback". Zero = no retry (default).
    fallback_retry_attempts: int = Field(default=0, ge=0, le=10)
    fallback_retry_delay_ms: int = Field(default=200, ge=0, le=5000)


class ContextManifest(BaseModel):
    chp_version: str = "0.1"
    agent_id: str
    task: str
    requires: ContextRequirements
    token_budget: int
    on_missing: Literal["fail_hard", "warn", "proceed", "ledger_fallback"] = "warn"
    # Case C: cross-session dependency — explicit parent session to fall back to.
    # Only queried after current-session fallback exhausts retries and still missing.
    # None (default) = no cross-session query (safe default — no data bleed).
    parent_session_id: str | None = None

    @classmethod
    def to_json_schema(cls) -> dict:
        schema = cls.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        return schema
