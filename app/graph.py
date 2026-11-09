"""LangGraph orchestration.

  retrieve ──► eligibility ──► planner ──► verify_costs ──► skeptic ──┐
                                  ▲                                  │ blocking issues &
                                  └──────────────────────────────────┘ progress possible
                                                                     │ otherwise
                                                                     ▼
                                                                  finalize
"""
from __future__ import annotations

import json
import uuid
from typing import TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agents.eligibility import run_eligibility
from app.agents.llm import call_llm
from app.agents.planner import deterministic_plan, llm_plan, update_memory
from app.agents.skeptic import critique, llm_advisories, monthly_debt_service, verify_plan_costs
from app.config import get_settings
from app.models import BusinessProfile, FundingPlan, Issue, PlanResponse
from app.rag.store import get_store
from app.tools.finance import dscr
from app.tracing import RequestTracer


class GraphState(TypedDict, total=False):
    profile: BusinessProfile
    retrieval: dict
    eligible: list[str]
    rejected: dict
    candidates: list[str]
    plan: FundingPlan
    issues: list[Issue]
    critique_history: list[dict]
    iteration: int
    memory: dict
    next_memory: dict
    stalled: bool
    final: dict


def _tracer(config: RunnableConfig) -> RequestTracer:
    return config["configurable"]["tracer"]


def _lc_config(config: RunnableConfig) -> dict:
    return {"callbacks": _tracer(config).callbacks()}


# ------------------------------------------------------------------- nodes
def retrieve_node(state: GraphState, config: RunnableConfig) -> dict:
    p = state["profile"]
    with _tracer(config).span("hierarchical_rag", "retriever", input=p.model_dump()) as h:
        r = get_store().retrieve(p)
        h["output"] = {"schemes": r["scheme_ids"], "chunks": len(r["citations"])}
        h["output_full"] = {"metrics": r["metrics"],
                            "citations": [c.model_dump() for c in r["citations"]]}
    return {"retrieval": r}


def eligibility_node(state: GraphState, config: RunnableConfig) -> dict:
    p, r = state["profile"], state["retrieval"]
    tr = _tracer(config)
    with tr.span("eligibility_agent", "agent", input={"retrieved": r["scheme_ids"]}) as h:
        eligible, rejected = run_eligibility(p)
        candidates = [s for s in r["scheme_ids"] if s in eligible and s != "cgtmse"]
        lending = [s for s in eligible if s != "cgtmse"]
        recall = len(candidates) / len(lending) if lending else 1.0
        r["metrics"]["eligible_recall"] = round(recall, 3)
        r["metrics"]["retrieval_precision"] = round(
            len(candidates) / len(r["scheme_ids"]), 3) if r["scheme_ids"] else 0.0
        h["output"] = {"eligible": eligible, "candidates": candidates,
                       "rejected": list(rejected)}
    tr.score("retrieval_eligible_recall", recall, "share of rule-eligible schemes surfaced by RAG")
    tr.score("retrieval_level2_mean_similarity", r["metrics"]["level2_mean_score"])
    return {"eligible": eligible, "rejected": rejected, "candidates": candidates,
            "iteration": 0, "memory": {}, "critique_history": [], "issues": []}


def planner_node(state: GraphState, config: RunnableConfig) -> dict:
    s = get_settings()
    p = state["profile"]
    it = state["iteration"] + 1
    mem = state.get("next_memory", state["memory"]) if it > 1 else {}
    with _tracer(config).span("planner_agent", "agent",
                              input={"iteration": it, "memory": mem,
                                     "critique": [i.model_dump() for i in state["issues"]]}) as h:
        plan, source = None, "deterministic"
        if s.llm_enabled:
            plan = llm_plan(p, state["candidates"], state["issues"], state.get("plan"),
                            s.min_dscr, _lc_config(config))
            source = "llm" if plan else "deterministic_fallback"
        if plan is None:
            plan = deterministic_plan(p, state["candidates"], mem)
        h["output"] = {"source": source, "debt": plan.debt, "subsidy": plan.subsidy,
                       "items": [(li.scheme_id, li.amount, li.tenure_months, li.with_cgtmse)
                                 for li in plan.line_items]}
    return {"plan": plan, "iteration": it, "memory": mem}


def verify_costs_node(state: GraphState, config: RunnableConfig) -> dict:
    with _tracer(config).span("cost_verifier_tool", "tool",
                              input=[li.scheme_id for li in state["plan"].line_items]) as h:
        plan, calls = verify_plan_costs(state["plan"].model_copy(deep=True))
        h["output"] = {"tool_calls": len(calls),
                       "monthly_debt_service": monthly_debt_service(plan)}
        h["output_full"] = calls
    return {"plan": plan}


def skeptic_node(state: GraphState, config: RunnableConfig) -> dict:
    s = get_settings()
    p, plan = state["profile"], state["plan"]
    with _tracer(config).span("skeptic_agent", "evaluator", input={"iteration": state["iteration"]}) as h:
        issues = critique(p, plan, state["eligible"], s.min_dscr, s.nbfc_share_warning,
                          s.cost_tolerance_pct)
        blocking = [i for i in issues if i.severity == "blocking"]
        if s.llm_enabled and not blocking:
            issues += llm_advisories(p, plan, issues, _lc_config(config))
        next_mem = update_memory(state["memory"], issues)
        progress = s.llm_enabled or {k: v for k, v in next_mem.items() if k != "dscr_exhausted"} != \
            {k: v for k, v in state["memory"].items() if k != "dscr_exhausted"}
        service = monthly_debt_service(plan)
        ratio = dscr(p.monthly_cash_flow, service)
        entry = {
            "iteration": state["iteration"],
            "plan": [{"scheme_id": li.scheme_id, "amount": li.amount, "tenure_months": li.tenure_months,
                      "with_cgtmse": li.with_cgtmse, "emi": li.emi} for li in plan.line_items],
            "monthly_debt_service": service,
            "dscr": round(ratio, 3) if ratio else None,
            "total_cost_of_credit": round(sum(li.total_cost_of_credit for li in plan.line_items), 2),
            "blocking": [f"{i.code}: {i.message}" for i in blocking],
            "warnings": [f"{i.code}: {i.message}" for i in issues if i.severity == "warning"],
            "verdict": "revise" if blocking and progress and state["iteration"] < s.max_iterations
            else ("accept" if not blocking else "escalate"),
        }
        h["output"] = {"verdict": entry["verdict"], "blocking": entry["blocking"],
                       "dscr": entry["dscr"]}
    return {"issues": issues, "next_memory": next_mem, "stalled": not progress,
            "critique_history": state["critique_history"] + [entry]}


def route_after_skeptic(state: GraphState) -> str:
    return "planner" if state["critique_history"][-1]["verdict"] == "revise" else "finalize"


def _template_summary(p: BusinessProfile, plan: FundingPlan, status: str, ratio, gap, cost,
                      apr, iterations, open_issues) -> str:
    parts = [f"{p.business_name}: Rs {p.project_cost:,.0f} {p.purpose.replace('_', ' ')} project in "
             f"{p.sector}."]
    src = [f"own contribution Rs {plan.owner_contribution:,.0f}"]
    if plan.subsidy:
        src.append(f"PMEGP subsidy Rs {plan.subsidy:,.0f}")
    for li in plan.line_items:
        extra = " + CGTMSE cover" if li.with_cgtmse else ""
        src.append(f"{li.scheme_name} Rs {li.amount:,.0f} @ {li.rate_pct}% for "
                   f"{li.tenure_months} months{extra}")
    parts.append("Recommended mix: " + "; ".join(src) + ".")
    if plan.line_items:
        parts.append(f"Total monthly debt service Rs {sum(li.emi for li in plan.line_items):,.0f}"
                     f"{f' (DSCR {ratio:.2f})' if ratio else ''}; verified total cost of credit "
                     f"Rs {cost:,.0f}, weighted effective APR {apr:.2f}%.")
    if iterations > 1:
        parts.append(f"The skeptic agent sent the plan back {iterations - 1} time(s) before "
                     f"{'accepting' if status != 'needs_attention' else 'escalating'} it.")
    if gap > 1:
        parts.append(f"Unfunded gap: Rs {gap:,.0f}.")
    blocking = [i.message for i in open_issues if i.severity == "blocking"]
    warnings = [i.message for i in open_issues if i.severity == "warning"]
    if blocking:
        parts.append("Needs attention: " + " ".join(blocking))
    if warnings:
        parts.append("Watch-outs: " + " ".join(warnings))
    return " ".join(parts)


def finalize_node(state: GraphState, config: RunnableConfig) -> dict:
    s = get_settings()
    tr = _tracer(config)
    p, plan, issues = state["profile"], state["plan"], state["issues"]
    with tr.span("report_writer", "agent") as h:
        blocking = [i for i in issues if i.severity == "blocking"]
        status = "needs_attention" if blocking else (
            "approved_with_warnings" if issues else "approved")
        service = monthly_debt_service(plan)
        ratio = dscr(p.monthly_cash_flow, service)
        cost = round(sum(li.total_cost_of_credit for li in plan.line_items), 2)
        apr = round(sum(li.effective_apr_pct * li.amount for li in plan.line_items) / plan.debt, 3) \
            if plan.debt else 0.0
        gap = round(max(p.project_cost - plan.total_funding, 0), 2)
        summary = _template_summary(p, plan, status, ratio, gap, cost, apr, state["iteration"], issues)
        if s.llm_enabled:
            raw = call_llm(
                "You are LoanScout's report writer. Rewrite the draft as a clear advisory summary "
                "for an Indian small-business owner in under 170 words. Keep every number exactly "
                "as given. Do not invent figures. Plain prose, no markdown headings.",
                json.dumps({"draft": summary, "open_issues": [i.message for i in issues]}),
                _lc_config(config))
            if raw and not raw.startswith("__LLM_ERROR__"):
                summary = raw.strip()
        h["output"] = {"status": status, "dscr": ratio, "cost": cost}
    tr.score("final_dscr", ratio or 0.0)
    tr.score("self_correction_iterations", state["iteration"])
    tr.score("open_blocking_issues", len(blocking))
    return {"final": {
        "status": status, "summary": summary, "monthly_debt_service": service,
        "dscr": round(ratio, 3) if ratio else None, "total_cost_of_credit": cost,
        "weighted_apr_pct": apr, "funding_gap": gap,
    }}


def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("eligibility", eligibility_node)
    g.add_node("planner", planner_node)
    g.add_node("verify_costs", verify_costs_node)
    g.add_node("skeptic", skeptic_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eligibility")
    g.add_edge("eligibility", "planner")
    g.add_edge("planner", "verify_costs")
    g.add_edge("verify_costs", "skeptic")
    g.add_conditional_edges("skeptic", route_after_skeptic,
                            {"planner": "planner", "finalize": "finalize"})
    g.add_edge("finalize", END)
    return g.compile()


_GRAPH = None


def run_advisor(profile: BusinessProfile) -> PlanResponse:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    s = get_settings()
    rid = uuid.uuid4().hex[:12]
    with RequestTracer(rid) as tr:
        tr.set_io(input=profile.model_dump())
        state = _GRAPH.invoke({"profile": profile},
                              config={"configurable": {"tracer": tr}, "recursion_limit": 50})
        f = state["final"]
        resp = PlanResponse(
            request_id=rid, status=f["status"], summary=f["summary"], plan=state["plan"],
            project_cost=profile.project_cost, total_funding=round(state["plan"].total_funding, 2),
            funding_gap=f["funding_gap"], monthly_debt_service=f["monthly_debt_service"],
            dscr=f["dscr"], total_cost_of_credit=f["total_cost_of_credit"],
            weighted_apr_pct=f["weighted_apr_pct"], iterations=state["iteration"],
            critique_history=state["critique_history"],
            open_issues=state["issues"], eligible_schemes=state["eligible"],
            rejected_schemes=state["rejected"], citations=state["retrieval"]["citations"][:12],
            retrieval_metrics=state["retrieval"]["metrics"], trace=tr.events,
            mode="llm" if s.llm_enabled else "deterministic",
        )
        tr.set_io(output={"status": resp.status, "summary": resp.summary})
        resp.trace_url = tr.url
    return resp
