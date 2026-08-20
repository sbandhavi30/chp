"""
CHP PII detection — semantic pre-filter for chunk pools.

Two backends, same interface:

    RegexPIIFilter    — zero deps, 15 built-in patterns (SSN, credit card, email, etc.)
    PresidioPIIFilter — Microsoft Presidio (pip install "chp[pii]"), 50+ entity types,
                        ML-powered, language-aware

Usage — global hook (applies to every select_chunks call):

    from chp.pii import RegexPIIFilter, set_pii_filter
    set_pii_filter(RegexPIIFilter())          # keyword + regex

    # or Presidio
    from chp.pii import PresidioPIIFilter, set_pii_filter
    set_pii_filter(PresidioPIIFilter())

Usage — per-call (override global):

    selected = select_chunks(chunks, manifest, embedder, pii_filter=my_filter)

Both filters return True if a chunk contains PII — chunk is then excluded
before any routing, scoring, or ledger write. The existing keyword `exclude`
list still fires too; PII filter is additive, not a replacement.
"""
from __future__ import annotations

import logging
import re
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)

# ── Global PII filter registry ────────────────────────────────────────────────

_pii_filter: "PIIFilter | None" = None


def set_pii_filter(f: "PIIFilter | None") -> None:
    """Register a global PII filter applied to every select_chunks() call.

    Pass None to deregister.
    """
    global _pii_filter
    _pii_filter = f


def get_pii_filter() -> "PIIFilter | None":
    return _pii_filter


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class PIIFilter(Protocol):
    def contains_pii(self, text: str) -> bool:
        """Return True if text contains PII that should be excluded."""
        ...


# ── Regex patterns ─────────────────────────────────────────────────────────────
# Covers the most common PII types seen in multi-agent AI pipelines.
# Patterns are compiled once at import time.

_PATTERNS: dict[str, re.Pattern] = {
    # US SSN — 123-45-6789 or 123456789 or "Social Security"
    "ssn": re.compile(
        r"\b(?:\d{3}-\d{2}-\d{4}|\d{9})\b"
        r"|social\s+security(?:\s+number|\s+no\.?|\s+#)?",
        re.I,
    ),
    # Credit / debit card — Visa, MC, Amex, Discover (Luhn not checked — FP ok for security)
    "credit_card": re.compile(
        r"\b(?:4\d{3}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}"   # Visa
        r"|5[1-5]\d{2}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}"  # MC
        r"|3[47]\d{2}[\s\-]?\d{6}[\s\-]?\d{5}"                # Amex
        r"|6(?:011|5\d{2})[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b",  # Discover
        re.I,
    ),
    # Email address
    "email": re.compile(
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
    ),
    # US phone — (555) 123-4567 / 555-123-4567 / +1 555 123 4567
    "phone_us": re.compile(
        r"(?:\+1[\s\-]?)?"
        r"(?:\(\d{3}\)|\d{3})[\s\-]?\d{3}[\s\-]?\d{4}\b"
    ),
    # International phone — E.164 +44 7700 900000
    "phone_intl": re.compile(r"\+\d{1,3}[\s\-]?\d{4,14}\b"),
    # IP address — v4
    "ipv4": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    # IPv6 (abbreviated form too)
    "ipv6": re.compile(
        r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"
    ),
    # Date of birth patterns — "DOB:", "date of birth", "born on"
    "dob": re.compile(
        r"\b(?:dob|date[\s_]of[\s_]birth|born\s+on)\b",
        re.I,
    ),
    # Passport number — heuristic (letter(s) + 6-9 digits)
    "passport": re.compile(
        r"\b(?:passport(?:\s+(?:no|number|#))?[\s:]+)?[A-Z]{1,2}\d{6,9}\b",
        re.I,
    ),
    # Driver's license — varies by state/country, catch "license number" label
    "drivers_license": re.compile(
        r"\b(?:driver(?:\'?s)?[\s_]licen[sc]e|dl|d\.l\.)[\s:#]*[A-Z0-9\-]{5,20}\b",
        re.I,
    ),
    # Bank account / IBAN
    "bank_account": re.compile(
        r"\b(?:account[\s_](?:no|number|#)[\s:]*\d{6,20}"
        r"|iban[\s:]+[A-Z]{2}\d{2}[A-Z0-9]{4,30})\b",
        re.I,
    ),
    # Routing number label
    "routing_number": re.compile(
        r"\brouting[\s_](?:no|number|#)[\s:]*\d{9}\b",
        re.I,
    ),
    # Medical record number
    "mrn": re.compile(
        r"\b(?:mrn|medical[\s_]record[\s_](?:no|number|#))[\s:]*[A-Z0-9\-]{4,20}\b",
        re.I,
    ),
    # API key / token heuristic — high-entropy strings after key/token/secret/bearer labels
    "api_key": re.compile(
        r"\b(?:api[\s_]?key|token|secret|bearer|authorization)[\s:\"\']+[A-Za-z0-9+/=\-_]{20,}\b",
        re.I,
    ),
    # Password label
    "password": re.compile(
        r"\b(?:password|passwd|pwd)[\s:\"\'=]+\S{4,}",
        re.I,
    ),
}

# Entity labels for logging
_LABEL: dict[str, str] = {
    "ssn":             "SSN",
    "credit_card":     "CREDIT_CARD",
    "email":           "EMAIL",
    "phone_us":        "PHONE",
    "phone_intl":      "PHONE",
    "ipv4":            "IP_ADDRESS",
    "ipv6":            "IP_ADDRESS",
    "dob":             "DATE_OF_BIRTH",
    "passport":        "PASSPORT",
    "drivers_license": "DRIVERS_LICENSE",
    "bank_account":    "BANK_ACCOUNT",
    "routing_number":  "ROUTING_NUMBER",
    "mrn":             "MEDICAL_RECORD",
    "api_key":         "API_KEY",
    "password":        "PASSWORD",
}


class RegexPIIFilter:
    """
    Zero-dependency PII filter using compiled regex patterns.

    Covers 15 PII entity types. Faster than Presidio; lower recall on
    unusual formats. Combine with keyword `exclude` list for belt-and-suspenders.

    Args:
        enabled_types: Subset of entity types to check. None = all 15.
                       Available: ssn, credit_card, email, phone_us, phone_intl,
                       ipv4, ipv6, dob, passport, drivers_license, bank_account,
                       routing_number, mrn, api_key, password
        log_detections: If True (default), log entity type on match (not content).

    Example:
        f = RegexPIIFilter(enabled_types=["ssn", "credit_card", "email"])
        f.contains_pii("SSN: 123-45-6789")  # → True
        f.contains_pii("Hello world")        # → False
    """

    def __init__(
        self,
        enabled_types: list[str] | None = None,
        log_detections: bool = True,
    ) -> None:
        if enabled_types is not None:
            unknown = set(enabled_types) - _PATTERNS.keys()
            if unknown:
                raise ValueError(f"Unknown PII types: {unknown}. Valid: {sorted(_PATTERNS)}")
            self._patterns = {k: v for k, v in _PATTERNS.items() if k in enabled_types}
        else:
            self._patterns = _PATTERNS

        self._log = log_detections

    def contains_pii(self, text: str) -> bool:
        for name, pattern in self._patterns.items():
            if pattern.search(text):
                if self._log:
                    log.info(
                        json_safe_log("chp.pii.detected", {
                            "filter": "regex", "entity_type": _LABEL[name],
                        })
                    )
                return True
        return False

    def detected_types(self, text: str) -> list[str]:
        """Return list of entity type labels found in text (for debugging)."""
        return [
            _LABEL[name]
            for name, pattern in self._patterns.items()
            if pattern.search(text)
        ]


# ── Presidio backend ──────────────────────────────────────────────────────────

class PresidioPIIFilter:
    """
    PII filter backed by Microsoft Presidio — 50+ entity types, ML-powered.

    Install:  pip install "chp[pii]"
    Requires: presidio-analyzer, presidio-anonymizer, spacy en_core_web_lg

    Setup:
        python -m spacy download en_core_web_lg

    Args:
        entities:  List of Presidio entity types to detect. None = all.
                   e.g. ["PERSON", "EMAIL_ADDRESS", "CREDIT_CARD", "US_SSN"]
        language:  Language code. Default "en".
        score_threshold: Minimum confidence score (0.0–1.0). Default 0.5.
        log_detections: Log entity types on detection. Default True.

    Example:
        from chp.pii import PresidioPIIFilter, set_pii_filter
        set_pii_filter(PresidioPIIFilter(entities=["US_SSN", "CREDIT_CARD", "EMAIL_ADDRESS"]))
    """

    def __init__(
        self,
        entities: list[str] | None = None,
        language: str = "en",
        score_threshold: float = 0.5,
        log_detections: bool = True,
    ) -> None:
        try:
            from presidio_analyzer import AnalyzerEngine
        except ImportError as exc:
            raise ImportError(
                'presidio-analyzer is required for PresidioPIIFilter. '
                'Install with:  pip install "chp[pii]"'
            ) from exc

        self._analyzer = AnalyzerEngine()
        self._entities = entities
        self._language = language
        self._threshold = score_threshold
        self._log = log_detections

    def contains_pii(self, text: str) -> bool:
        results = self._analyzer.analyze(
            text=text,
            entities=self._entities,
            language=self._language,
            score_threshold=self._threshold,
        )
        if results:
            if self._log:
                types = list({r.entity_type for r in results})
                log.info(
                    json_safe_log("chp.pii.detected", {
                        "filter": "presidio", "entity_types": types,
                    })
                )
            return True
        return False

    def detected_types(self, text: str) -> list[str]:
        results = self._analyzer.analyze(
            text=text,
            entities=self._entities,
            language=self._language,
            score_threshold=self._threshold,
        )
        return list({r.entity_type for r in results})


# ── helpers ───────────────────────────────────────────────────────────────────

import json as _json

def json_safe_log(event: str, data: dict) -> str:
    return _json.dumps({"chp_event": event, **data})
