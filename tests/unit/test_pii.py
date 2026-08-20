"""
Unit tests for chp.pii — RegexPIIFilter, PresidioPIIFilter stub, global registry,
and integration with select_chunks.
"""
from __future__ import annotations

import pytest

from chp.pii import (
    RegexPIIFilter,
    PIIFilter,
    set_pii_filter,
    get_pii_filter,
)
from chp.schema.context_manifest import ContextManifest, ContextRequirements
from chp.schema.rationale_envelope import AnnotatedChunk
from chp.engine.scorer import select_chunks
from chp.engine.embedder import StubEmbedder
from chp.observability import CHPEvent


# ── RegexPIIFilter — per-type positive matches ────────────────────────────────

class TestRegexPIIFilterPatterns:
    def setup_method(self):
        self.f = RegexPIIFilter(log_detections=False)

    def test_ssn_dashes(self):
        assert self.f.contains_pii("SSN: 123-45-6789")

    def test_ssn_no_dashes(self):
        assert self.f.contains_pii("social security 123456789")

    def test_ssn_label(self):
        assert self.f.contains_pii("Social Security Number on file")

    def test_credit_card_visa(self):
        assert self.f.contains_pii("card: 4111 1111 1111 1111")

    def test_credit_card_amex(self):
        assert self.f.contains_pii("amex 3782 822463 10005")

    def test_email(self):
        assert self.f.contains_pii("reach me at user@example.com please")

    def test_phone_us(self):
        assert self.f.contains_pii("call (555) 123-4567")

    def test_phone_intl(self):
        assert self.f.contains_pii("intl: +44 7700 900000")

    def test_ipv4(self):
        assert self.f.contains_pii("server at 192.168.1.100")

    def test_ipv6(self):
        assert self.f.contains_pii("addr 2001:db8:85a3:0000:1319:8a2e:0370:7344")

    def test_dob(self):
        assert self.f.contains_pii("Date of Birth: 1990-01-01")

    def test_passport(self):
        assert self.f.contains_pii("passport A1234567")

    def test_bank_account(self):
        assert self.f.contains_pii("account number 123456789012")

    def test_routing_number(self):
        assert self.f.contains_pii("routing number 021000021")

    def test_mrn(self):
        assert self.f.contains_pii("MRN: ABC-12345")

    def test_api_key(self):
        assert self.f.contains_pii("api_key: sk-AbCdEfGhIjKlMnOpQrStUvWx")

    def test_password(self):
        assert self.f.contains_pii("password: s3cr3tP@ss!")

    def test_clean_text_false(self):
        assert not self.f.contains_pii("Hello, the weather is nice today.")

    def test_clean_number_not_ssn(self):
        # 8-digit number should not match SSN (needs exactly 9)
        assert not self.f.contains_pii("order 12345678 confirmed")


# ── RegexPIIFilter — enabled_types subset ────────────────────────────────────

class TestRegexPIIFilterSubset:
    def test_only_email_hits_email(self):
        f = RegexPIIFilter(enabled_types=["email"], log_detections=False)
        assert f.contains_pii("user@example.com")

    def test_only_email_misses_ssn(self):
        f = RegexPIIFilter(enabled_types=["email"], log_detections=False)
        assert not f.contains_pii("SSN: 123-45-6789")

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown PII types"):
            RegexPIIFilter(enabled_types=["not_a_real_type"])

    def test_detected_types_returns_labels(self):
        f = RegexPIIFilter(log_detections=False)
        types = f.detected_types("email me at x@y.com and SSN 123-45-6789")
        assert "EMAIL" in types
        assert "SSN" in types

    def test_detected_types_empty_clean_text(self):
        f = RegexPIIFilter(log_detections=False)
        assert f.detected_types("Hello world") == []


# ── PIIFilter protocol check ──────────────────────────────────────────────────

def test_regex_filter_satisfies_protocol():
    f = RegexPIIFilter(log_detections=False)
    assert isinstance(f, PIIFilter)


# ── Global registry ───────────────────────────────────────────────────────────

class TestGlobalRegistry:
    def setup_method(self):
        set_pii_filter(None)  # reset before each test

    def teardown_method(self):
        set_pii_filter(None)  # always clean up

    def test_default_none(self):
        assert get_pii_filter() is None

    def test_set_and_get(self):
        f = RegexPIIFilter(log_detections=False)
        set_pii_filter(f)
        assert get_pii_filter() is f

    def test_deregister_with_none(self):
        set_pii_filter(RegexPIIFilter(log_detections=False))
        set_pii_filter(None)
        assert get_pii_filter() is None


# ── Integration with select_chunks ────────────────────────────────────────────

def _manifest(exclude=None):
    return ContextManifest(
        agent_id="test-agent",
        task="test",
        requires=ContextRequirements(exclude=exclude or []),
        token_budget=10000,
    )


def _chunk(cid, content, token_cost=10):
    return AnnotatedChunk(
        chunk_id=cid,
        content=content,
        token_cost=token_cost,
        source_agent="src",
        source_turn=0,
    )


class TestSelectChunksPIIIntegration:
    def setup_method(self):
        set_pii_filter(None)

    def teardown_method(self):
        set_pii_filter(None)

    def test_pii_chunk_excluded_via_global_filter(self):
        set_pii_filter(RegexPIIFilter(log_detections=False))
        chunks = [
            _chunk("clean", "This is safe content about billing"),
            _chunk("pii", "SSN: 123-45-6789"),
        ]
        selected = select_chunks(chunks, _manifest(), StubEmbedder())
        ids = [c.chunk_id for c in selected]
        assert "clean" in ids
        assert "pii" not in ids

    def test_pii_chunk_excluded_via_per_call_filter(self):
        # global is None; per-call filter should still work
        f = RegexPIIFilter(log_detections=False)
        chunks = [
            _chunk("clean", "approved payment"),
            _chunk("pii", "password: hunter2"),
        ]
        selected = select_chunks(chunks, _manifest(), StubEmbedder(), pii_filter=f)
        ids = [c.chunk_id for c in selected]
        assert "clean" in ids
        assert "pii" not in ids

    def test_per_call_overrides_global(self):
        # global filter would hit email; per-call filter only checks ssn
        set_pii_filter(RegexPIIFilter(log_detections=False))
        per_call = RegexPIIFilter(enabled_types=["ssn"], log_detections=False)
        chunks = [
            _chunk("email_chunk", "contact: user@example.com"),
            _chunk("ssn_chunk", "SSN: 987-65-4321"),
        ]
        selected = select_chunks(chunks, _manifest(), StubEmbedder(), pii_filter=per_call)
        ids = [c.chunk_id for c in selected]
        # only ssn_chunk filtered; email_chunk passes because per_call only checks ssn
        assert "email_chunk" in ids
        assert "ssn_chunk" not in ids

    def test_chunk_excluded_event_has_pii_reason(self, capsys):
        events = []

        import chp
        chp.set_metrics_hook(lambda event, data: events.append((event, data)))

        f = RegexPIIFilter(log_detections=False)
        chunks = [_chunk("pii", "SSN: 123-45-6789")]
        select_chunks(chunks, _manifest(), StubEmbedder(), pii_filter=f)

        chp.set_metrics_hook(None)

        excluded = [e for e in events if e[0] == CHPEvent.CHUNK_EXCLUDED]
        assert len(excluded) == 1
        assert excluded[0][1]["reason"] == "pii"
        assert excluded[0][1]["chunk_id"] == "pii"

    def test_keyword_exclude_reason_field(self):
        events = []

        import chp
        chp.set_metrics_hook(lambda event, data: events.append((event, data)))

        manifest = _manifest(exclude=["forbidden"])
        chunks = [_chunk("kw", "this text contains forbidden word")]
        select_chunks(chunks, manifest, StubEmbedder())

        chp.set_metrics_hook(None)

        excluded = [e for e in events if e[0] == CHPEvent.CHUNK_EXCLUDED]
        assert len(excluded) == 1
        assert excluded[0][1]["reason"] == "keyword"

    def test_no_filter_no_exclusion(self):
        # Without any filter, SSN text passes through (keyword exclude is empty)
        chunks = [_chunk("ssn", "SSN: 123-45-6789")]
        selected = select_chunks(chunks, _manifest(), StubEmbedder())
        assert any(c.chunk_id == "ssn" for c in selected)


# ── PresidioPIIFilter import guard ────────────────────────────────────────────

def test_presidio_import_error_without_install():
    """PresidioPIIFilter raises ImportError with install hint if presidio not installed."""
    import sys
    import importlib

    # Temporarily make presidio_analyzer unimportable
    real_presidio = sys.modules.get("presidio_analyzer")
    sys.modules["presidio_analyzer"] = None  # type: ignore[assignment]

    try:
        from chp.pii import PresidioPIIFilter
        with pytest.raises(ImportError, match='pip install "chp\\[pii\\]"'):
            PresidioPIIFilter()
    finally:
        if real_presidio is None:
            sys.modules.pop("presidio_analyzer", None)
        else:
            sys.modules["presidio_analyzer"] = real_presidio
