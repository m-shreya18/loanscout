"""Pydantic schemas shared by the API, the graph and the agents."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Sector = Literal["manufacturing", "services", "trading"]
Purpose = Literal["setup", "machinery", "working_capital", "expansion"]
Category = Literal["woman", "sc_st", "obc", "minority", "ex_serviceman", "pwd", "general"]


class BusinessProfile(BaseModel):
    business_name: str = "My Business"
    sector: Sector
    purpose: Purpose
    project_cost: float = Field(gt=0, description="Total project cost in INR")
    owner_contribution: float = Field(ge=0, description="Promoter's own money in INR")
    monthly_cash_flow: float = Field(
        gt=0, description="Projected monthly net cash flow available for debt service (INR)"
    )
    annual_revenue: float = Field(default=0, ge=0)
    business_age_years: float = Field(default=0, ge=0)
    is_greenfield: bool = True
    collateral_value: float = Field(default=0, ge=0)
    credit_score: int = Field(default=700, ge=300, le=900)
    owner_categories: list[Category] = Field(default_factory=lambda: ["general"])
    location: Literal["urban", "rural"] = "urban"
    sells_to_large_buyers: bool = False
    goal: str = Field(default="", description="Free-text description of what the money is for")

    @model_validator(mode="after")
    def _check(self):
        if self.owner_contribution > self.project_cost:
            raise ValueError("owner_contribution cannot exceed project_cost")
        if self.is_greenfield and self.business_age_years > 1:
            raise ValueError("a greenfield project cannot have business_age_years > 1")
        return self

    @property
    def is_special_category(self) -> bool:
        return any(c != "general" for c in self.owner_categories)


class LineItem(BaseModel):
    scheme_id: str
    scheme_name: str
    amount: float
    rate_pct: float
    tenure_months: int
    moratorium_months: int = 0
    with_cgtmse: bool = False
    # Filled by the cost-verification tool
    emi: float = 0.0
    total_interest: float = 0.0
    processing_fee: float = 0.0
    guarantee_fee: float = 0.0
    total_cost_of_credit: float = 0.0
    effective_apr_pct: float = 0.0
    claimed_total_cost: float | None = None  # what the planner said (LLM mode) before verification


class Issue(BaseModel):
    code: str
    severity: Literal["blocking", "warning"]
    message: str
    scheme_id: str | None = None
    suggestion: str | None = None


class FundingPlan(BaseModel):
    line_items: list[LineItem] = Field(default_factory=list)
    owner_contribution: float = 0.0
    subsidy: float = 0.0
    subsidy_scheme: str | None = None
    rationale: str = ""

    @property
    def debt(self) -> float:
        return sum(li.amount for li in self.line_items)

    @property
    def total_funding(self) -> float:
        return self.debt + self.owner_contribution + self.subsidy


class Citation(BaseModel):
    scheme_id: str
    section: str
    text: str
    score: float


class PlanResponse(BaseModel):
    request_id: str
    status: Literal["approved", "approved_with_warnings", "needs_attention"]
    summary: str
    plan: FundingPlan
    project_cost: float
    total_funding: float
    funding_gap: float
    monthly_debt_service: float
    dscr: float | None
    total_cost_of_credit: float
    weighted_apr_pct: float
    iterations: int
    critique_history: list[dict]
    open_issues: list[Issue]
    eligible_schemes: list[str]
    rejected_schemes: dict[str, list[str]]
    citations: list[Citation]
    retrieval_metrics: dict
    trace: list[dict]
    trace_url: str | None = None
    mode: Literal["llm", "deterministic"]
