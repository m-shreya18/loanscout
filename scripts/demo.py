"""Run the advisor on the sample scenarios and print a readable report.
Usage: python -m scripts.demo [scenario_name]"""
import sys

from app.graph import run_advisor
from app.models import BusinessProfile
from scripts.scenarios import SCENARIOS


def show(name: str):
    r = run_advisor(BusinessProfile(**SCENARIOS[name]))
    print("=" * 100)
    print(f"{name}  →  {r.status.upper()}  (mode={r.mode}, iterations={r.iterations})")
    print("-" * 100)
    for h in r.critique_history:
        print(f"  iter {h['iteration']}: DSCR={h['dscr']}  cost=Rs {h['total_cost_of_credit']:,.0f}  "
              f"verdict={h['verdict']}")
        for b in h["blocking"]:
            print(f"      ✗ {b}")
    print("  Final plan:")
    print(f"    Own contribution           Rs {r.plan.owner_contribution:>12,.0f}")
    if r.plan.subsidy:
        print(f"    PMEGP subsidy              Rs {r.plan.subsidy:>12,.0f}")
    for li in r.plan.line_items:
        print(f"    {li.scheme_name[:26]:26} Rs {li.amount:>12,.0f}  {li.rate_pct:5.2f}%  "
              f"{li.tenure_months:3d}m  EMI {li.emi:>9,.0f}  cost {li.total_cost_of_credit:>10,.0f}  "
              f"APR {li.effective_apr_pct:5.2f}%{'  +CGTMSE' if li.with_cgtmse else ''}")
    print(f"  Funding Rs {r.total_funding:,.0f} / {r.project_cost:,.0f}   gap Rs {r.funding_gap:,.0f}   "
          f"debt service Rs {r.monthly_debt_service:,.0f}/m   DSCR {r.dscr}")
    print(f"  Retrieval: {r.retrieval_metrics['level1_hits']}")
    print(f"     eligible_recall={r.retrieval_metrics['eligible_recall']}  "
          f"precision={r.retrieval_metrics['retrieval_precision']}  "
          f"L2 mean sim={r.retrieval_metrics['level2_mean_score']}")
    for i in r.open_issues:
        print(f"  [{i.severity}] {i.code}: {i.message}")
    print(f"  Summary: {r.summary}")


if __name__ == "__main__":
    names = sys.argv[1:] or list(SCENARIOS)
    for n in names:
        show(n)
