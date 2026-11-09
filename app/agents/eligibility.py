"""Eligibility agent: rule checks of a business profile against every scheme."""
from __future__ import annotations

from app.models import BusinessProfile
from app.rag.knowledge import load_schemes


def check_scheme(p: BusinessProfile, s: dict) -> list[str]:
    """Return the list of reasons a scheme does NOT fit (empty list = eligible)."""
    el = s["eligibility"]
    reasons = []
    if p.sector not in el["sectors"]:
        reasons.append(f"sector '{p.sector}' not covered")
    if p.purpose not in el["purposes"]:
        reasons.append(f"purpose '{p.purpose}' not covered")
    if el["greenfield_only"] and not p.is_greenfield:
        reasons.append("only for new (greenfield) units")
    if p.business_age_years < el["min_business_age_years"]:
        reasons.append(f"needs {el['min_business_age_years']}+ years of business vintage")
    if el["categories_any"] and not set(el["categories_any"]) & set(p.owner_categories):
        reasons.append(f"reserved for {' / '.join(el['categories_any'])} entrepreneurs")
    if p.credit_score < el["min_credit_score"]:
        reasons.append(f"credit score {p.credit_score} below {el['min_credit_score']}")
    flag = s.get("requires_flag")
    if flag and not getattr(p, flag, False):
        reasons.append(f"requires {flag.replace('_', ' ')}")

    min_contrib = el["min_owner_contribution_pct"]
    if s["id"] == "pmegp":
        min_contrib = 5 if p.is_special_category else 10
        cap = s["project_cost_cap"].get(p.sector, 0)
        if p.project_cost > cap:
            reasons.append(f"project cost above PMEGP cap of Rs {cap:,.0f} for {p.sector}")
    if s["instrument"] != "guarantee" and p.owner_contribution < p.project_cost * min_contrib / 100:
        reasons.append(f"needs at least {min_contrib}% owner contribution")
    # The loan floor must be reachable for this project.
    if s["instrument"] != "guarantee":
        max_fundable = p.project_cost * s.get("max_project_share", 1.0)
        if max_fundable < s["min_amount"]:
            reasons.append(f"project too small for minimum ticket of Rs {s['min_amount']:,.0f}")
    return reasons


def run_eligibility(p: BusinessProfile) -> tuple[list[str], dict[str, list[str]]]:
    eligible, rejected = [], {}
    for sid, s in load_schemes().items():
        r = check_scheme(p, s)
        if r:
            rejected[sid] = r
        else:
            eligible.append(sid)
    return eligible, rejected


def pmegp_subsidy(p: BusinessProfile) -> float:
    s = load_schemes()["pmegp"]
    band = "special" if p.is_special_category else "general"
    return p.project_cost * s["subsidy_pct"][band][p.location] / 100
