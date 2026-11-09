"""Planner agent: drafts and revises a funding plan.

Deterministic mode: greedy fill of the funding need with the cheapest eligible instruments,
then apply the skeptic's critique through a small "planner memory" (attach CGTMSE, stretch
tenure, prefer low-EMI lenders, exclude a scheme, trim over-funding).

LLM mode: Gemini drafts the plan as JSON from the candidates, fee tables and critique. The
output is schema-validated and clamped; anything unusable falls back to deterministic mode.
"""
from __future__ import annotations

import json

from app.agents.eligibility import pmegp_subsidy
from app.agents.llm import call_llm, extract_json
from app.models import BusinessProfile, FundingPlan, Issue, LineItem
from app.rag.knowledge import fee_table, load_schemes
from app.tools.finance import REVOLVING, emi

EXCLUSIVE_GROUPS = {"mudra": {"mudra_shishu", "mudra_kishore", "mudra_tarun"}}
GOVT_BONUS = {"government_scheme": -0.75, "government_subsidy": -0.75}
DRAFT_TENURE = 36  # first draft minimises total interest; the skeptic pushes back if EMI is too high


def group_of(scheme_id: str) -> str | None:
    for g, members in EXCLUSIVE_GROUPS.items():
        if scheme_id in members:
            return g
    return None


def pick_rate(s: dict, credit_score: int) -> float:
    lo, hi = s["interest_rate"]["min"], s["interest_rate"]["max"]
    if credit_score >= 780:
        f = 0.15
    elif credit_score >= 740:
        f = 0.35
    elif credit_score >= 700:
        f = 0.5
    else:
        f = 0.8
    return round(lo + (hi - lo) * f, 2)


def _rank_key(s: dict, p: BusinessProfile, mem: dict) -> float:
    """Approximate annual cost of a rupee borrowed (lower is better)."""
    rate = pick_rate(s, p.credit_score)
    years = s["max_tenure_months"] / 12
    fee = s["fees"].get("processing_fee_pct", 0) / max(years, 1)
    guarantee = 0.5 if s["collateral_required"] and p.collateral_value <= 0 else 0
    key = rate + fee + guarantee + GOVT_BONUS.get(s["provider_type"], 0)
    if mem.get("prefer_low_emi") and s["instrument"] not in REVOLVING:
        # Monthly EMI per lakh at the longest allowed tenure dominates the ranking.
        key = emi(100000, rate, s["max_tenure_months"]) / 100
    return key


def deterministic_plan(p: BusinessProfile, candidates: list[str], mem: dict) -> FundingPlan:
    schemes = load_schemes()
    plan = FundingPlan(owner_contribution=p.owner_contribution)
    excluded = set(mem.get("excluded", []))
    pool = [c for c in candidates if c not in excluded and c != "cgtmse"]

    if "pmegp" in pool:
        plan.subsidy = round(pmegp_subsidy(p), 2)
        plan.subsidy_scheme = "pmegp"

    need = p.project_cost - p.owner_contribution - plan.subsidy
    initial_need = need
    used_groups: set[str] = set()
    ordered = sorted(pool, key=lambda sid: _rank_key(schemes[sid], p, mem))
    # PMEGP's subsidy only flows through the PMEGP loan, so finance that first.
    if "pmegp" in ordered:
        ordered.remove("pmegp")
        ordered.insert(0, "pmegp")

    for sid in ordered:
        if need <= 1:
            break
        s = schemes[sid]
        g = group_of(sid)
        if g and g in used_groups:
            continue
        cap = min(s["max_amount"], p.project_cost * s.get("max_project_share", 1.0))
        amount = min(need, cap)
        if amount < s["min_amount"]:
            continue
        # Avoid fragmenting a large requirement into tiny loans (each with its own paperwork),
        # unless that small loan would close the remaining need.
        if amount < need and cap < 0.2 * initial_need:
            continue
        revolving = s["instrument"] in REVOLVING
        if revolving:
            tenure, mora = s["max_tenure_months"], 0
        elif mem.get("stretch_tenure") or mem.get("prefer_low_emi"):
            tenure = s["max_tenure_months"]
            mora = mem.get("moratorium", {}).get(sid, 0)
        else:
            tenure, mora = min(DRAFT_TENURE, s["max_tenure_months"]), 0
        plan.line_items.append(LineItem(
            scheme_id=sid, scheme_name=s["name"], amount=round(amount, 2),
            rate_pct=pick_rate(s, p.credit_score), tenure_months=tenure,
            moratorium_months=mora, with_cgtmse=sid in mem.get("cgtmse_for", []),
        ))
        need -= amount
        if g:
            used_groups.add(g)

    notes = []
    if plan.subsidy:
        notes.append(f"PMEGP margin-money subsidy of Rs {plan.subsidy:,.0f} reduces the loan.")
    if mem.get("cgtmse_for"):
        notes.append("CGTMSE cover replaces collateral on: " + ", ".join(mem["cgtmse_for"]) + ".")
    if mem.get("prefer_low_emi"):
        notes.append("Lenders re-ranked by lowest EMI to protect cash flow.")
    elif mem.get("stretch_tenure"):
        notes.append("Tenures stretched to scheme maximum to bring EMIs within cash flow.")
    plan.rationale = " ".join(notes) or "Cheapest eligible instruments first, government schemes preferred."
    return plan


def update_memory(mem: dict, issues: list[Issue]) -> dict:
    """Translate skeptic critique into concrete planning adjustments."""
    mem = {**mem, "excluded": list(mem.get("excluded", [])), "cgtmse_for": list(mem.get("cgtmse_for", []))}
    for i in issues:
        if i.severity != "blocking":
            continue
        if i.code == "COLLATERAL_SHORTFALL" and i.scheme_id:
            if i.suggestion == "attach_cgtmse" and i.scheme_id not in mem["cgtmse_for"]:
                mem["cgtmse_for"].append(i.scheme_id)
            elif i.suggestion != "attach_cgtmse" and i.scheme_id not in mem["excluded"]:
                mem["excluded"].append(i.scheme_id)
        elif i.code == "DSCR_LOW":
            if not mem.get("stretch_tenure"):
                mem["stretch_tenure"] = True
            elif not mem.get("prefer_low_emi"):
                mem["prefer_low_emi"] = True
            else:
                mem["dscr_exhausted"] = True
        elif i.code in {"INELIGIBLE_SCHEME", "DUPLICATE_GROUP"} and i.scheme_id:
            if i.scheme_id not in mem["excluded"]:
                mem["excluded"].append(i.scheme_id)
    return mem


# ------------------------------------------------------------------------ LLM mode
PLANNER_SYSTEM = """You are the Planner agent of LoanScout, an Indian MSME lending advisor.
Build a funding plan that covers the project cost at the lowest total cost of credit while
keeping DSCR (monthly cash flow / monthly debt service) at or above {min_dscr}.
Rules:
- Use ONLY scheme_ids from the candidate list, within each scheme's amount range, max tenure,
  interest range and max_project_share of project cost.
- At most one Mudra loan. Collateral-required loans need collateral or with_cgtmse=true.
- If 'pmegp' is used, set use_pmegp_subsidy=true.
- Address every blocking issue in the critique.
Respond with JSON only:
{{"use_pmegp_subsidy": bool, "rationale": str,
  "line_items": [{{"scheme_id": str, "amount": number, "rate_pct": number,
                   "tenure_months": int, "moratorium_months": int, "with_cgtmse": bool,
                   "claimed_total_cost": number}}]}}"""


def llm_plan(p: BusinessProfile, candidates: list[str], critique: list[Issue],
             previous: FundingPlan | None, min_dscr: float, config: dict | None) -> FundingPlan | None:
    schemes = load_schemes()
    tables = {sid: fee_table(sid) for sid in candidates}
    user = json.dumps({
        "profile": p.model_dump(),
        "candidates": {sid: {"name": schemes[sid]["name"], "instrument": schemes[sid]["instrument"],
                             **tables[sid]} for sid in candidates},
        "pmegp_subsidy_if_used": pmegp_subsidy(p) if "pmegp" in candidates else 0,
        "previous_plan": previous.model_dump() if previous else None,
        "critique": [i.model_dump() for i in critique],
    }, default=str)
    raw = call_llm(PLANNER_SYSTEM.format(min_dscr=min_dscr), user, config)
    data = extract_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get("line_items"), list):
        return None
    plan = FundingPlan(owner_contribution=p.owner_contribution,
                       rationale=str(data.get("rationale", ""))[:1200])
    for li in data["line_items"]:
        sid = li.get("scheme_id")
        if sid not in schemes or sid == "cgtmse":
            continue
        try:
            plan.line_items.append(LineItem(
                scheme_id=sid, scheme_name=schemes[sid]["name"], amount=float(li["amount"]),
                rate_pct=float(li["rate_pct"]), tenure_months=int(li["tenure_months"]),
                moratorium_months=int(li.get("moratorium_months", 0) or 0),
                with_cgtmse=bool(li.get("with_cgtmse", False)),
                claimed_total_cost=float(li["claimed_total_cost"]) if li.get("claimed_total_cost") else None,
            ))
        except (KeyError, TypeError, ValueError):
            continue
    if data.get("use_pmegp_subsidy") and "pmegp" in candidates and any(
        li.scheme_id == "pmegp" for li in plan.line_items
    ):
        plan.subsidy = round(pmegp_subsidy(p), 2)
        plan.subsidy_scheme = "pmegp"
    return plan if plan.line_items else None
