from chp.templates import ManifestTemplates
from chp.schema.context_manifest import ContextManifest


def test_billing_template_returns_manifest():
    m = ManifestTemplates.billing_agent()
    assert isinstance(m, ContextManifest)
    assert "order_id" in m.requires.must_carry
    assert "user_id" in m.requires.must_carry
    assert m.on_missing == "fail_hard"
    assert "PII_raw" in m.requires.exclude


def test_auth_template_excludes_pii():
    m = ManifestTemplates.auth_agent()
    assert "PII_raw" in m.requires.exclude
    assert "auth_status" in m.requires.must_carry


def test_compliance_template_keeps_pii_flags():
    m = ManifestTemplates.compliance_agent()
    # compliance agents need PII flags — should NOT exclude PII_raw
    assert "PII_raw" not in m.requires.exclude
    assert "compliance_flags" in m.requires.must_carry


def test_token_budget_override():
    m = ManifestTemplates.billing_agent(token_budget=500)
    assert m.token_budget == 500


def test_list_templates_returns_names():
    names = ManifestTemplates.list_templates()
    assert "billing_agent" in names
    assert "auth_agent" in names
    assert len(names) >= 8


def test_all_templates_have_domain_tags():
    for name in ManifestTemplates.list_templates():
        m = getattr(ManifestTemplates, name)()
        assert len(m.requires.domain_tags) >= 3, f"{name} has too few domain_tags"
