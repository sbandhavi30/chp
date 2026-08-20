import pytest
from chp.schema.context_manifest import ContextManifest, ContextRequirements


def test_manifest_valid_construction():
    req = ContextRequirements(
        must_carry=["user_id", "order_id"],
        domain_tags=["billing", "auth"],
        history_depth="decisions_only",
        recency_window="last_3_agent_turns",
        exclude=["PII_raw"],
    )
    m = ContextManifest(
        chp_version="0.1",
        agent_id="billing-agent-v2",
        task="resolve_refund_dispute",
        requires=req,
        token_budget=2000,
        on_missing="fail_hard",
    )
    assert m.agent_id == "billing-agent-v2"
    assert m.requires.must_carry == ["user_id", "order_id"]
    assert m.on_missing == "fail_hard"


def test_manifest_rejects_invalid_on_missing():
    with pytest.raises(Exception):
        ContextManifest(
            chp_version="0.1",
            agent_id="x",
            task="y",
            requires=ContextRequirements(
                must_carry=[], domain_tags=[], history_depth="full",
                recency_window=None, exclude=[]
            ),
            token_budget=1000,
            on_missing="explode",
        )


def test_manifest_json_schema_export():
    schema = ContextManifest.to_json_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "properties" in schema
    assert "agent_id" in schema["properties"]
    assert "token_budget" in schema["properties"]
