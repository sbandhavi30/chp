from __future__ import annotations
from chp.schema.context_manifest import ContextManifest, ContextRequirements

# Common PII fields — always excluded by default unless overridden
_PII_FIELDS = ["PII_raw", "credit_card", "ssn", "dob", "passport", "bank_account"]
_NOISE_FIELDS = ["debug_trace", "stack_trace", "verbose_log"]
_DEFAULT_EXCLUDE = _PII_FIELDS + _NOISE_FIELDS


class ManifestTemplates:
    """
    Pre-built ContextManifest templates for common agent archetypes.
    Each returns a fully configured manifest the developer can use as-is
    or customise with .model_copy(update={...}).
    """

    @staticmethod
    def billing_agent(token_budget: int = 2000) -> ContextManifest:
        return ContextManifest(
            agent_id="billing-agent",
            task="resolve_billing_dispute",
            requires=ContextRequirements(
                must_carry=["order_id", "user_id"],
                domain_tags=["billing", "order", "charge", "refund", "payment", "duplicate"],
                history_depth="decisions_only",
                exclude=_DEFAULT_EXCLUDE,
            ),
            token_budget=token_budget,
            on_missing="fail_hard",
        )

    @staticmethod
    def auth_agent(token_budget: int = 500) -> ContextManifest:
        return ContextManifest(
            agent_id="auth-agent",
            task="verify_customer_identity",
            requires=ContextRequirements(
                must_carry=["user_id", "auth_status"],
                domain_tags=["auth", "identity", "session", "mfa", "verification"],
                history_depth="decisions_only",
                exclude=_DEFAULT_EXCLUDE,
            ),
            token_budget=token_budget,
            on_missing="warn",
        )

    @staticmethod
    def summarizer_agent(token_budget: int = 1000) -> ContextManifest:
        return ContextManifest(
            agent_id="summarizer-agent",
            task="summarize_resolution",
            requires=ContextRequirements(
                must_carry=["order_id"],
                domain_tags=["summary", "resolution", "refund", "customer_request", "outcome"],
                history_depth="summary",
                exclude=_DEFAULT_EXCLUDE + ["auth_status", "session_token", "fraud_score"],
            ),
            token_budget=token_budget,
            on_missing="warn",
        )

    @staticmethod
    def compliance_agent(token_budget: int = 800) -> ContextManifest:
        return ContextManifest(
            agent_id="compliance-agent",
            task="check_regulatory_compliance",
            requires=ContextRequirements(
                must_carry=["user_id", "compliance_flags"],
                domain_tags=["compliance", "GDPR", "PCI", "SOX", "regulatory", "data_residency"],
                history_depth="full",
                exclude=_NOISE_FIELDS,  # compliance NEEDS PII flags — don't exclude them
            ),
            token_budget=token_budget,
            on_missing="fail_hard",
        )

    @staticmethod
    def fraud_agent(token_budget: int = 600) -> ContextManifest:
        return ContextManifest(
            agent_id="fraud-agent",
            task="assess_fraud_risk",
            requires=ContextRequirements(
                must_carry=["fraud_score"],
                domain_tags=["fraud", "risk", "velocity", "device", "ip", "anomaly"],
                history_depth="decisions_only",
                exclude=_DEFAULT_EXCLUDE,
            ),
            token_budget=token_budget,
            on_missing="warn",
        )

    @staticmethod
    def research_agent(token_budget: int = 3000) -> ContextManifest:
        return ContextManifest(
            agent_id="research-agent",
            task="research_and_retrieve",
            requires=ContextRequirements(
                must_carry=[],
                domain_tags=["research", "facts", "knowledge", "sources", "context"],
                history_depth="full",
                exclude=_DEFAULT_EXCLUDE,
            ),
            token_budget=token_budget,
            on_missing="proceed",
        )

    @staticmethod
    def orchestrator(token_budget: int = 4000) -> ContextManifest:
        return ContextManifest(
            agent_id="orchestrator",
            task="synthesize_and_decide",
            requires=ContextRequirements(
                must_carry=[],
                domain_tags=["summary", "result", "decision", "outcome", "resolution"],
                history_depth="full",
                exclude=_DEFAULT_EXCLUDE,
            ),
            token_budget=token_budget,
            on_missing="proceed",
        )

    @staticmethod
    def policy_agent(token_budget: int = 800) -> ContextManifest:
        return ContextManifest(
            agent_id="policy-agent",
            task="check_policy_eligibility",
            requires=ContextRequirements(
                must_carry=["order_id"],
                domain_tags=["policy", "eligibility", "terms", "refund_policy", "rules"],
                history_depth="decisions_only",
                exclude=_DEFAULT_EXCLUDE,
            ),
            token_budget=token_budget,
            on_missing="fail_hard",
        )

    @staticmethod
    def support_router(token_budget: int = 500) -> ContextManifest:
        return ContextManifest(
            agent_id="support-router",
            task="classify_and_route_ticket",
            requires=ContextRequirements(
                must_carry=["user_id"],
                domain_tags=["routing", "classification", "intent", "ticket", "triage"],
                history_depth="none",
                exclude=_DEFAULT_EXCLUDE,
            ),
            token_budget=token_budget,
            on_missing="warn",
        )

    @staticmethod
    def code_reviewer(token_budget: int = 5000) -> ContextManifest:
        return ContextManifest(
            agent_id="code-reviewer",
            task="review_code_changes",
            requires=ContextRequirements(
                must_carry=[],
                domain_tags=["code", "diff", "review", "security", "style", "bug"],
                history_depth="decisions_only",
                exclude=_NOISE_FIELDS,
            ),
            token_budget=token_budget,
            on_missing="proceed",
        )

    @staticmethod
    def list_templates() -> list[str]:
        return [
            "billing_agent", "auth_agent", "summarizer_agent", "compliance_agent",
            "fraud_agent", "research_agent", "orchestrator", "policy_agent",
            "support_router", "code_reviewer",
        ]
