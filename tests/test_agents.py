import json

import pytest

from app.agents.eligibility import run_eligibility
from app.config import get_settings
from app.graph import run_advisor
from app.models import BusinessProfile


def test_eligibility_rules(scenarios):
    eligible, rejected = run_eligibility(BusinessProfile(**scenarios["existing_cnc_expansion"]))
    assert "bank_msme_term_loan" in eligible
    assert "pmegp" in rejected and any("greenfield" in r for r in rejected["pmegp"])
    assert "standup_india" in rejected  # neither woman nor SC/ST, and not greenfield


def test_greenfield_woman_uses_subsidy_and_balances(scenarios):
    r = run_advisor(BusinessProfile(**scenarios["woman_bakery_greenfield"]))
    assert r.status.startswith("approved")
    assert r.plan.subsidy > 0
    assert r.funding_gap == 0
    assert r.dscr >= 1.25


def test_self_correction_fixes_collateral_and_dscr(scenarios):
    r = run_advisor(BusinessProfile(**scenarios["existing_cnc_expansion"]))
    first = r.critique_history[0]
    assert any(b.startswith("COLLATERAL_SHORTFALL") for b in first["blocking"])
    assert any(b.startswith("DSCR_LOW") for b in first["blocking"])
    assert r.iterations >= 2 and r.status.startswith("approved")
    assert all(li.with_cgtmse for li in r.plan.line_items if li.scheme_id == "bank_msme_term_loan")
    assert r.dscr >= 1.25


def test_unaffordable_project_is_escalated_not_faked(scenarios):
    r = run_advisor(BusinessProfile(**scenarios["overleveraged_cafe"]))
    assert r.status == "needs_attention"
    assert any(i.code == "DSCR_LOW" for i in r.open_issues)
    assert r.critique_history[-1]["verdict"] == "escalate"


def test_every_scenario_produces_verified_costs(scenarios):
    for name, s in scenarios.items():
        r = run_advisor(BusinessProfile(**s))
        assert r.plan.line_items, name
        for li in r.plan.line_items:
            assert li.total_cost_of_credit > 0 and li.emi > 0, name
        steps = [t["step"] for t in r.trace if t["step"] != "score"]
        assert steps[:3] == ["hierarchical_rag", "eligibility_agent", "planner_agent"], name
        assert "cost_verifier_tool" in steps and "skeptic_agent" in steps


def test_skeptic_catches_hallucinated_llm_plan(monkeypatch, scenarios):
    """Simulate Gemini: round 1 invents a rate below the scheme floor, overstates the loan and
    under-reports cost; round 2 complies. The skeptic must reject round 1."""
    import app.agents.planner as planner
    import app.graph as graph

    calls = {"n": 0}

    def fake_llm(system, user, config=None):
        if "Planner" not in system:
            return json.dumps({"notes": ["Keep GST filings current for the bank's review."]})
        calls["n"] += 1
        if calls["n"] == 1:
            items = [{"scheme_id": "pmegp", "amount": 2_000_000, "rate_pct": 7.0, "tenure_months": 36,
                      "moratorium_months": 0, "with_cgtmse": False, "claimed_total_cost": 50_000}]
        else:
            items = [{"scheme_id": "pmegp", "amount": 1_575_000, "rate_pct": 10.7, "tenure_months": 60,
                      "moratorium_months": 6, "with_cgtmse": False, "claimed_total_cost": 520_000}]
        return json.dumps({"use_pmegp_subsidy": True, "rationale": "test", "line_items": items})

    monkeypatch.setenv("USE_LLM", "true")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    get_settings.cache_clear()
    monkeypatch.setattr(planner, "call_llm", fake_llm)
    monkeypatch.setattr("app.agents.skeptic.call_llm", fake_llm)
    monkeypatch.setattr(graph, "call_llm", lambda *a, **k: None)
    try:
        r = run_advisor(BusinessProfile(**scenarios["woman_bakery_greenfield"]))
    finally:
        get_settings.cache_clear()
    first = r.critique_history[0]["blocking"]
    assert any(b.startswith("RATE_OUT_OF_RANGE") for b in first)
    assert any(b.startswith("OVER_FUNDED") for b in first)
    assert r.iterations == 2 and r.mode == "llm"
    assert r.status.startswith("approved")
    # Verified cost replaces the planner's claim; mismatch reported if >1% off.
    assert r.plan.line_items[0].total_cost_of_credit > 0
    assert any(i.code == "ADVISORY" for i in r.open_issues)
