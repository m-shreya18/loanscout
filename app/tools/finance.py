"""Deterministic cost-of-credit tools. The skeptic uses these to verify every number
the planner proposes — the LLM is never trusted with arithmetic."""
from __future__ import annotations

from dataclasses import dataclass


def emi(principal: float, annual_rate_pct: float, months: int) -> float:
    """Standard reducing-balance EMI."""
    if principal <= 0 or months <= 0:
        return 0.0
    r = annual_rate_pct / 1200
    if r == 0:
        return principal / months
    f = (1 + r) ** months
    return principal * r * f / (f - 1)


def amortization(principal: float, annual_rate_pct: float, months: int, moratorium: int = 0):
    """Yield (month, opening_balance, payment, interest, principal_paid).
    During the moratorium only interest is paid; the EMI tenure follows it."""
    r = annual_rate_pct / 1200
    bal = principal
    m = 0
    for _ in range(moratorium):
        m += 1
        interest = bal * r
        yield m, bal, interest, interest, 0.0
    pay = emi(principal, annual_rate_pct, months)
    for _ in range(months):
        m += 1
        interest = bal * r
        p = min(pay - interest, bal)
        yield m, bal, interest + p, interest, p
        bal -= p


def guarantee_fee_pct(scheme: dict, amount: float) -> float:
    slabs = scheme["fees"].get("guarantee_fee_slabs")
    if not slabs:
        return scheme["fees"].get("guarantee_fee_pct_annual", 0.0)
    for s in slabs:
        if amount <= s["upto"]:
            return s["pct"]
    return slabs[-1]["pct"]


def processing_fee(scheme: dict, amount: float) -> float:
    f = scheme["fees"]
    fee = amount * f.get("processing_fee_pct", 0) / 100
    cap = f.get("processing_fee_cap", 0)
    if cap:
        fee = min(fee, cap)
    return fee * (1 + f.get("gst_on_fees_pct", 0) / 100)


def _irr_monthly(flows: list[float]) -> float:
    """Bisection IRR on monthly cash flows (flows[0] is the net amount received)."""
    lo, hi = 0.0, 0.2

    def npv(r):
        return sum(cf / (1 + r) ** i for i, cf in enumerate(flows))

    if npv(lo) * npv(hi) > 0:
        return 0.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


@dataclass
class CostBreakdown:
    emi: float
    total_interest: float
    processing_fee: float
    guarantee_fee: float
    total_cost_of_credit: float
    effective_apr_pct: float

    def as_dict(self):
        return {k: round(v, 2) for k, v in self.__dict__.items()}


def verify_cost(
    scheme: dict,
    amount: float,
    rate_pct: float,
    tenure_months: int,
    moratorium_months: int = 0,
    cgtmse_scheme: dict | None = None,
) -> CostBreakdown:
    """Full cost of one loan: interest + processing fee (with GST) + annual guarantee fee
    charged on the outstanding balance at the start of each year, plus effective APR."""
    schedule = list(amortization(amount, rate_pct, tenure_months, moratorium_months))
    total_interest = sum(row[3] for row in schedule)
    pfee = processing_fee(scheme, amount)

    gfee_pct = 0.0
    if cgtmse_scheme is not None:
        gfee_pct = guarantee_fee_pct(cgtmse_scheme, amount)
    elif scheme["fees"].get("guarantee_fee_pct_annual"):
        gfee_pct = scheme["fees"]["guarantee_fee_pct_annual"]

    gfee_by_month = {}
    if gfee_pct:
        for month, opening, *_ in schedule:
            if (month - 1) % 12 == 0:
                gfee_by_month[month] = opening * gfee_pct / 100
    gfee_total = sum(gfee_by_month.values())

    # Effective APR: net disbursal vs every outflow (instalments + guarantee fees).
    flows = [amount - pfee - gfee_by_month.get(1, 0.0)]
    for month, _, payment, *_ in schedule:
        out = payment + gfee_by_month.get(month + 1, 0.0)
        flows.append(-out)
    apr = ((1 + _irr_monthly(flows)) ** 12 - 1) * 100 if amount > 0 else 0.0

    return CostBreakdown(
        emi=emi(amount, rate_pct, tenure_months),
        total_interest=total_interest,
        processing_fee=pfee,
        guarantee_fee=gfee_total,
        total_cost_of_credit=total_interest + pfee + gfee_total,
        effective_apr_pct=apr,
    )


REVOLVING = {"working_capital", "invoice_discounting"}


def verify_revolving_cost(
    scheme: dict, amount: float, rate_pct: float, months: int = 12,
    cgtmse_scheme: dict | None = None,
) -> CostBreakdown:
    """Working-capital lines are interest-only and rolled over: cost one year of use.
    `emi` here is the monthly interest servicing."""
    months = max(months, 1)
    interest = amount * rate_pct / 100 * months / 12
    pfee = processing_fee(scheme, amount) + scheme["fees"].get("platform_fee_annual", 0)
    gfee = amount * guarantee_fee_pct(cgtmse_scheme, amount) / 100 if cgtmse_scheme else 0.0
    total = interest + pfee + gfee
    apr = total / amount * 12 / months * 100 if amount else 0.0
    return CostBreakdown(
        emi=amount * rate_pct / 1200, total_interest=interest, processing_fee=pfee,
        guarantee_fee=gfee, total_cost_of_credit=total, effective_apr_pct=apr,
    )


def max_sustainable_loan(monthly_budget: float, annual_rate_pct: float, months: int) -> float:
    """Inverse EMI: the largest principal whose EMI fits the budget."""
    r = annual_rate_pct / 1200
    if r == 0:
        return monthly_budget * months
    return monthly_budget * ((1 + r) ** months - 1) / (r * (1 + r) ** months)


def dscr(monthly_cash_flow: float, monthly_debt_service: float) -> float | None:
    if monthly_debt_service <= 0:
        return None
    return monthly_cash_flow / monthly_debt_service
