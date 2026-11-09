import pytest

from app.rag.knowledge import get_scheme
from app.tools.finance import (amortization, emi, max_sustainable_loan, processing_fee,
                               verify_cost)


def test_emi_matches_textbook_value():
    # Rs 10 lakh, 12% p.a., 5 years → Rs 22,244.45 per month
    assert emi(1_000_000, 12, 60) == pytest.approx(22244.45, abs=0.01)


def test_amortization_repays_principal_exactly():
    rows = list(amortization(500_000, 11, 48, moratorium=6))
    assert len(rows) == 54
    assert sum(r[4] for r in rows) == pytest.approx(500_000, abs=0.01)
    assert all(r[4] == 0 for r in rows[:6])  # interest-only during moratorium


def test_processing_fee_cap_and_gst():
    kishore = get_scheme("mudra_kishore")  # 0.5% + 18% GST
    assert processing_fee(kishore, 100_000) == pytest.approx(500 * 1.18)
    bank = get_scheme("bank_msme_term_loan")  # 1% capped at Rs 2 lakh
    assert processing_fee(bank, 30_000_000) == pytest.approx(200_000 * 1.18)


def test_effective_apr_exceeds_nominal_when_fees_exist():
    nbfc = get_scheme("nbfc_unsecured")
    c = verify_cost(nbfc, 1_000_000, 20, 36)
    assert c.effective_apr_pct > 20 + 1.0
    assert c.total_cost_of_credit == pytest.approx(c.total_interest + c.processing_fee)


def test_cgtmse_guarantee_fee_added():
    bank = get_scheme("bank_msme_term_loan")
    plain = verify_cost(bank, 3_000_000, 10, 84)
    covered = verify_cost(bank, 3_000_000, 10, 84, cgtmse_scheme=get_scheme("cgtmse"))
    assert covered.guarantee_fee > 0
    assert covered.total_cost_of_credit > plain.total_cost_of_credit


def test_max_sustainable_loan_inverts_emi():
    p = max_sustainable_loan(25_000, 11, 84)
    assert emi(p, 11, 84) == pytest.approx(25_000, rel=1e-6)
