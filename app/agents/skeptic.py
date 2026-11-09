"""Cost verifier tool node + Skeptic agent.

The verifier recomputes every line with the deterministic finance tools. The skeptic then
attacks the plan: eligibility, limits, collateral, coverage, DSCR, cost claims, and expensive
debt. Blocking issues send the plan back to the planner (self-correction loop)."""
from __future__ import annotations

import json

from app.agents.eligibility import check_scheme
from app.agents.llm import call_llm, extract_json
from app.agents.planner import group_of
from app.models import BusinessProfile, FundingPlan, Issue
from app.rag.knowledge import load_schemes
from app.tools.finance import REVOLVING, dscr, max_sustainable_loan, verify_cost, verify_revolving_cost


def verify_plan_costs(plan: FundingPlan) -> tuple[FundingPlan, list[dict]]:
    """Fill verified cost fields; return per-line tool-call records for tracing."""
    schemes = load_schemes()
    cg = schemes["cgtmse"]
    calls = []
    for li in plan.line_items:
        s = schemes[li.scheme_id]
        if s["instrument"] in REVOLVING:
            c = verify_revolving_cost(s, li.amount, li.rate_pct, li.tenure_months,
                                      cg if li.with_cgtmse else None)
        else:
            c = verify_cost(s, li.amount, li.rate_pct, li.tenure_months, li.moratorium_months,
                            cg if li.with_cgtmse else None)
        li.emi = round(c.emi, 2)
        li.total_interest = round(c.total_interest, 2)
        li.processing_fee = round(c.processing_fee, 2)
        li.guarantee_fee = round(c.guarantee_fee, 2)
        li.total_cost_of_credit = round(c.total_cost_of_credit, 2)
        li.effective_apr_pct = round(c.effective_apr_pct, 3)
        calls.append({"tool": "verify_cost", "scheme_id": li.scheme_id,
                      "input": {"amount": li.amount, "rate_pct": li.rate_pct,
                                "tenure_months": li.tenure_months,
                                "moratorium_months": li.moratorium_months,
                                "with_cgtmse": li.with_cgtmse},
                      "output": c.as_dict()})
    return plan, calls


def monthly_debt_service(plan: FundingPlan) -> float:
    return round(sum(li.emi for li in plan.line_items), 2)


def critique(p: BusinessProfile, plan: FundingPlan, eligible: list[str], min_dscr: float,
             nbfc_warn: float, tol_pct: float) -> list[Issue]:
    schemes = load_schemes()
    issues: list[Issue] = []
    cg = schemes["cgtmse"]
    seen_groups: dict[str, str] = {}

    for li in plan.line_items:
        s = schemes[li.scheme_id]
        reasons = check_scheme(p, s)
        if li.scheme_id not in eligible or reasons:
            issues.append(Issue(code="INELIGIBLE_SCHEME", severity="blocking", scheme_id=li.scheme_id,
                                message=f"{s['name']}: " + ("; ".join(reasons) or "not in eligible set")))
        g = group_of(li.scheme_id)
        if g:
            if g in seen_groups:
                issues.append(Issue(code="DUPLICATE_GROUP", severity="blocking", scheme_id=li.scheme_id,
                                    message=f"Only one {g} loan is allowed per borrower "
                                            f"(already using {seen_groups[g]})."))
            seen_groups[g] = li.scheme_id
        cap = min(s["max_amount"], p.project_cost * s.get("max_project_share", 1.0))
        if li.amount > cap + 1 or li.amount < s["min_amount"] - 1:
            issues.append(Issue(code="LIMIT_EXCEEDED", severity="blocking", scheme_id=li.scheme_id,
                                message=f"{s['name']}: Rs {li.amount:,.0f} outside allowed "
                                        f"Rs {s['min_amount']:,.0f}–{cap:,.0f}."))
        lo, hi = s["interest_rate"]["min"], s["interest_rate"]["max"]
        if not lo - 0.01 <= li.rate_pct <= hi + 0.01:
            issues.append(Issue(code="RATE_OUT_OF_RANGE", severity="blocking", scheme_id=li.scheme_id,
                                message=f"{s['name']}: rate {li.rate_pct}% outside {lo}–{hi}%."))
        if li.tenure_months > s["max_tenure_months"] or li.moratorium_months > s["max_moratorium_months"]:
            issues.append(Issue(code="TENURE_EXCEEDED", severity="blocking", scheme_id=li.scheme_id,
                                message=f"{s['name']}: tenure/moratorium beyond scheme maximum."))
        if li.claimed_total_cost is not None and li.total_cost_of_credit > 0:
            diff = abs(li.claimed_total_cost - li.total_cost_of_credit) / li.total_cost_of_credit * 100
            if diff > tol_pct:
                issues.append(Issue(code="COST_MISMATCH", severity="warning", scheme_id=li.scheme_id,
                                    message=f"{s['name']}: planner claimed Rs {li.claimed_total_cost:,.0f}, "
                                            f"verified cost is Rs {li.total_cost_of_credit:,.0f} "
                                            f"({diff:.1f}% off). Verified figure adopted."))
                li.claimed_total_cost = None

    # Collateral: pledge available collateral to secured loans in order.
    remaining = p.collateral_value
    for li in plan.line_items:
        s = schemes[li.scheme_id]
        if not s["collateral_required"] or li.with_cgtmse:
            continue
        if remaining >= li.amount:
            remaining -= li.amount
            continue
        can_cover = li.amount <= cg["max_amount"] and p.sector in cg["eligibility"]["sectors"]
        issues.append(Issue(
            code="COLLATERAL_SHORTFALL", severity="blocking", scheme_id=li.scheme_id,
            message=f"{s['name']} needs collateral for Rs {li.amount:,.0f} but only "
                    f"Rs {max(remaining, 0):,.0f} is unpledged.",
            suggestion="attach_cgtmse" if can_cover else "drop_scheme",
        ))
        remaining = 0

    gap = p.project_cost - plan.total_funding
    if gap > max(1000, p.project_cost * 0.005):
        issues.append(Issue(code="FUNDING_GAP", severity="blocking",
                            message=f"Plan leaves Rs {gap:,.0f} of the project unfunded."))
    elif gap < -max(1000, p.project_cost * 0.005):
        issues.append(Issue(code="OVER_FUNDED", severity="blocking",
                            message=f"Plan raises Rs {-gap:,.0f} more than the project needs."))

    service = monthly_debt_service(plan)
    ratio = dscr(p.monthly_cash_flow, service)
    if ratio is not None and ratio < min_dscr:
        term = [li for li in plan.line_items if schemes[li.scheme_id]["instrument"] not in REVOLVING]
        budget = p.monthly_cash_flow / min_dscr - sum(
            li.emi for li in plan.line_items if li not in term)
        sustainable = sum(
            max_sustainable_loan(max(budget, 0) * li.emi / max(sum(t.emi for t in term), 1),
                                 li.rate_pct, schemes[li.scheme_id]["max_tenure_months"])
            for li in term
        )
        term_debt = sum(li.amount for li in term)
        fix = ("stretching tenures should fix this." if sustainable >= term_debt else
               f"roughly Rs {term_debt - sustainable:,.0f} must come from extra equity or a smaller project.")
        issues.append(Issue(
            code="DSCR_LOW", severity="blocking",
            message=f"DSCR {ratio:.2f} is below {min_dscr}: debt service Rs {service:,.0f}/month vs "
                    f"cash flow Rs {p.monthly_cash_flow:,.0f}/month. At maximum tenures the business "
                    f"can sustain about Rs {sustainable:,.0f} of term debt; {fix}",
        ))

    debt = plan.debt
    nbfc = sum(li.amount for li in plan.line_items if schemes[li.scheme_id]["provider_type"] == "nbfc")
    if debt and nbfc / debt > nbfc_warn:
        issues.append(Issue(code="EXPENSIVE_DEBT", severity="warning",
                            message=f"{nbfc / debt:.0%} of the debt is high-cost NBFC credit "
                                    f"(16–24%). Consider raising equity or building collateral to "
                                    f"move it to a bank or scheme loan."))
    for li in plan.line_items:
        if li.scheme_id == "treds_invoice_discounting":
            issues.append(Issue(code="RECEIVABLES_ONLY", severity="warning", scheme_id=li.scheme_id,
                                message=f"TReDS funds only invoices raised on TReDS-registered buyers; "
                                        f"confirm at least Rs {li.amount:,.0f} of such receivables per "
                                        f"cycle, otherwise move the inventory part to a cash-credit limit."))
        if schemes[li.scheme_id]["fees"].get("prepayment_penalty_pct", 0) >= 2 and li.tenure_months >= 60:
            issues.append(Issue(code="PREPAYMENT_PENALTY", severity="warning", scheme_id=li.scheme_id,
                                message=f"{li.scheme_name} charges up to "
                                        f"{schemes[li.scheme_id]['fees']['prepayment_penalty_pct']}% "
                                        f"on foreclosure — negotiate this before signing."))
    return issues


SKEPTIC_SYSTEM = """You are the Skeptic agent of LoanScout. Deterministic checks have already
verified eligibility, limits, collateral, DSCR and costs. Add at most 3 short, concrete,
NON-numeric risk notes a careful credit officer would raise about this plan (documentation,
concentration, timing, subsidy release conditions, etc.). Do not repeat existing issues.
Respond with JSON only: {"notes": [str, ...]}"""


def llm_advisories(p: BusinessProfile, plan: FundingPlan, issues: list[Issue],
                   config: dict | None) -> list[Issue]:
    raw = call_llm(SKEPTIC_SYSTEM, json.dumps({
        "profile": p.model_dump(), "plan": plan.model_dump(),
        "existing_issues": [i.message for i in issues]}, default=str), config)
    data = extract_json(raw)
    if not isinstance(data, dict):
        return []
    return [Issue(code="ADVISORY", severity="warning", message=str(n)[:400])
            for n in data.get("notes", [])[:3]]
