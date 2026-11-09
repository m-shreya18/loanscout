from app.models import BusinessProfile
from app.rag.store import get_store


def test_hierarchical_retrieval_is_scheme_aware(scenarios):
    p = BusinessProfile(**scenarios["trader_working_capital"])
    r = get_store().retrieve(p)
    # Level-1 metadata filter: PMEGP excludes trading and must never appear.
    assert "pmegp" not in r["scheme_ids"]
    assert "treds_invoice_discounting" in r["scheme_ids"]
    # Level-2 chunks only come from schemes chosen at level 1 (+ CGTMSE fee table).
    allowed = set(r["scheme_ids"]) | {"cgtmse"}
    assert {c.scheme_id for c in r["citations"]} <= allowed
    assert {c.section for c in r["citations"]} & {"fees", "fee_table"}


def test_fee_questions_hit_fee_sections(scenarios):
    p = BusinessProfile(**scenarios["woman_bakery_greenfield"])
    r = get_store().retrieve(p)
    assert r["metrics"]["level2_chunks"] > 0
    assert 0 < r["metrics"]["section_coverage"] <= 1


def test_retrieval_surfaces_every_eligible_lending_scheme(scenarios):
    from app.agents.eligibility import run_eligibility
    for name, s in scenarios.items():
        p = BusinessProfile(**s)
        eligible, _ = run_eligibility(p)
        retrieved = get_store().retrieve(p)["scheme_ids"]
        missed = [e for e in eligible if e != "cgtmse" and e not in retrieved]
        assert not missed, f"{name}: retrieval missed {missed}"
