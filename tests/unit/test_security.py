"""
Security regression tests — verifies the 5 CRITICAL fixes hold.

Fix 1+2: ID sanitization (SQL injection prevention)
Fix 3:   Thread lock on concurrent writes
Fix 4:   json.loads fallback in LLM inference
Fix 5:   LLM output clamped before manifest creation
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from chp.inference import _infer_with_llm
from chp.ledger.lancedb_ledger import CHPLedger, _safe_id
from chp.schema.rationale_envelope import AnnotatedChunk, RationaleEnvelope


def _env(chunk_id: str) -> RationaleEnvelope:
    return RationaleEnvelope(
        chunk_id=chunk_id, content="content", source_agent="a", source_turn=1,
        hop_sequence=["a", "b"], selected_because=["test"], score=0.5,
        must_carry=False, token_cost=10,
    )


# ── Fix 1+2: _safe_id rejects injection payloads ─────────────────────────────

@pytest.mark.parametrize("bad_id", [
    "sess' OR '1'='1",           # classic SQL injection
    "sess; DROP TABLE chp_ledger",
    "sess\" OR \"1\"=\"1",
    "../../../etc/passwd",
    "sess\x00null",
    "sess AND hop_number=0",
    "",                          # empty
    "   ",                       # whitespace
])
def test_safe_id_rejects_injection(bad_id):
    with pytest.raises((ValueError, TypeError)):
        _safe_id(bad_id, "session_id")


@pytest.mark.parametrize("good_id", [
    "session-123",
    "agent_billing_v2",
    "sess:2026.08.19",
    "USR-8821",
    "demo-1787243920",
])
def test_safe_id_accepts_valid_ids(good_id):
    assert _safe_id(good_id) == good_id


def test_ledger_write_rejects_injected_session_id():
    ledger = CHPLedger()
    with pytest.raises(ValueError, match="invalid characters"):
        ledger.write("sess' OR '1'='1", 0, "a", "b", _env("c1"))


def test_ledger_write_rejects_injected_from_agent():
    ledger = CHPLedger()
    with pytest.raises(ValueError, match="invalid characters"):
        ledger.write("sess-ok", 0, "agent'; DROP TABLE chp_ledger--", "b", _env("c2"))


def test_ledger_query_rejects_injected_session_id():
    ledger = CHPLedger()
    with pytest.raises(ValueError, match="invalid characters"):
        ledger.query("sess' OR '1'='1")


def test_ledger_prune_rejects_injected_session_id():
    ledger = CHPLedger()
    with pytest.raises(ValueError, match="invalid characters"):
        ledger.prune("'; DELETE FROM chp_ledger WHERE '1'='1")


def test_ledger_chunk_id_injection_blocked():
    """Chunk with injected chunk_id must be rejected at write time."""
    ledger = CHPLedger()
    bad_env = RationaleEnvelope(
        chunk_id="c1' OR '1'='1",
        content="content", source_agent="a", source_turn=1,
        hop_sequence=["a", "b"], selected_because=[], score=0.5,
        must_carry=False, token_cost=10,
    )
    with pytest.raises(ValueError, match="invalid characters"):
        ledger.write("sess-ok", 0, "agent-a", "agent-b", bad_env)


# ── Fix 3: Thread lock — concurrent writes don't corrupt ─────────────────────

def test_concurrent_writes_no_data_corruption():
    """50 concurrent threads writing unique chunks to same session — all succeed, no loss."""
    ledger = CHPLedger()
    errors: list[str] = []
    session_id = "concurrent-stress-001"

    def write_chunk(i: int):
        try:
            env = _env(f"chunk-{i:04d}")
            ledger.write(session_id, i, f"agent-{i}", "orchestrator", env)
        except Exception as e:
            errors.append(f"thread-{i}: {e}")

    threads = [threading.Thread(target=write_chunk, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent write errors:\n" + "\n".join(errors)
    records = ledger.query(session_id)
    assert len(records) == 50, f"Expected 50 records, got {len(records)}"


def test_concurrent_duplicate_writes_raise_not_corrupt():
    """Same chunk written by 10 threads — exactly one succeeds, rest raise RuntimeError."""
    ledger = CHPLedger()
    session_id = "concurrent-dup-001"
    successes = []
    errors = []

    def write_same(i: int):
        try:
            ledger.write(session_id, 0, "agent-a", "agent-b", _env("shared-chunk"))
            successes.append(i)
        except RuntimeError:
            pass  # expected for duplicates
        except Exception as e:
            errors.append(str(e))

    threads = [threading.Thread(target=write_same, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Unexpected errors: {errors}"
    assert len(successes) == 1, f"Expected exactly 1 success, got {len(successes)}"
    records = ledger.query(session_id)
    assert len(records) == 1


# ── Fix 4: json.loads fallback ────────────────────────────────────────────────

def _mock_llm_client(response_text: str):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    client.chat.completions.create.return_value = MagicMock(choices=[choice])
    return client


def test_llm_inference_falls_back_on_invalid_json():
    """If LLM returns non-JSON, infer_manifest falls back to heuristic — no crash."""
    client = _mock_llm_client("Sorry, I cannot help with that.")
    from chp.inference import infer_manifest
    from chp.schema.context_manifest import ContextManifest
    m = infer_manifest(
        role="Billing Specialist",
        goal="Resolve duplicate charge disputes",
        llm_client=client,
    )
    assert isinstance(m, ContextManifest)
    assert m.token_budget > 0


def test_llm_inference_falls_back_on_partial_json():
    client = _mock_llm_client('{"must_carry": ["order_id"')  # truncated JSON
    from chp.inference import infer_manifest
    m = infer_manifest(role="Billing Specialist", goal="Refunds", llm_client=client)
    assert m.token_budget > 0


def test_llm_inference_falls_back_on_empty_response():
    client = _mock_llm_client("")
    from chp.inference import infer_manifest
    m = infer_manifest(role="Auth Specialist", goal="Verify identity", llm_client=client)
    assert m.token_budget > 0


# ── Fix 5: LLM output clamped ─────────────────────────────────────────────────

def test_llm_negative_token_budget_clamped_to_minimum():
    client = _mock_llm_client(json.dumps({
        "must_carry": ["order_id"],
        "domain_tags": ["billing"],
        "history_depth": "decisions_only",
        "exclude": ["PII_raw"],
        "token_budget": -999999,
        "on_missing": "warn",
        "reasoning": "test",
    }))
    from chp.inference import infer_manifest
    m = infer_manifest(role="Billing Specialist", goal="Refunds", llm_client=client)
    assert m.token_budget >= 100, f"token_budget not clamped: {m.token_budget}"


def test_llm_excessive_token_budget_clamped_to_maximum():
    client = _mock_llm_client(json.dumps({
        "must_carry": [], "domain_tags": ["x"],
        "history_depth": "full", "exclude": [],
        "token_budget": 99999999,
        "on_missing": "warn", "reasoning": "test",
    }))
    from chp.inference import infer_manifest
    m = infer_manifest(role="Research Agent", goal="Search", llm_client=client)
    assert m.token_budget <= 50000, f"token_budget not clamped: {m.token_budget}"


def test_llm_invalid_on_missing_defaults_to_warn():
    client = _mock_llm_client(json.dumps({
        "must_carry": [], "domain_tags": ["x"],
        "history_depth": "full", "exclude": [],
        "token_budget": 1000,
        "on_missing": "explode_everything",
        "reasoning": "test",
    }))
    from chp.inference import infer_manifest
    m = infer_manifest(role="Some Agent", goal="Do stuff", llm_client=client)
    assert m.on_missing == "warn"


def test_llm_invalid_history_depth_defaults_to_decisions_only():
    client = _mock_llm_client(json.dumps({
        "must_carry": [], "domain_tags": ["x"],
        "history_depth": "everything_forever",
        "exclude": [], "token_budget": 1000,
        "on_missing": "warn", "reasoning": "test",
    }))
    from chp.inference import infer_manifest
    m = infer_manifest(role="Some Agent", goal="Do stuff", llm_client=client)
    assert m.requires.history_depth == "decisions_only"


def test_llm_oversized_must_carry_list_clamped():
    client = _mock_llm_client(json.dumps({
        "must_carry": [f"field_{i}" for i in range(100)],
        "domain_tags": ["billing"], "history_depth": "full",
        "exclude": [], "token_budget": 1000,
        "on_missing": "warn", "reasoning": "test",
    }))
    from chp.inference import infer_manifest
    m = infer_manifest(role="Some Agent", goal="Do stuff", llm_client=client)
    assert len(m.requires.must_carry) <= 20
