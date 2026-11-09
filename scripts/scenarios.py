"""Sample business profiles used by the demo script and the tests."""
SCENARIOS = {
    "woman_bakery_greenfield": dict(
        business_name="Meera's Bakehouse", sector="manufacturing", purpose="setup",
        project_cost=2_500_000, owner_contribution=300_000, monthly_cash_flow=75_000,
        is_greenfield=True, business_age_years=0, collateral_value=0, credit_score=745,
        owner_categories=["woman"], location="urban",
        goal="Set up a commercial bakery with ovens, mixers and a delivery van"),
    "existing_cnc_expansion": dict(
        business_name="Shakti Precision Works", sector="manufacturing", purpose="machinery",
        project_cost=4_000_000, owner_contribution=900_000, monthly_cash_flow=95_000,
        is_greenfield=False, business_age_years=6, collateral_value=1_500_000, credit_score=760,
        owner_categories=["general"], location="urban",
        goal="Buy two CNC machines to take on a new auto-parts order"),
    "trader_working_capital": dict(
        business_name="Gupta Electricals", sector="trading", purpose="working_capital",
        project_cost=800_000, owner_contribution=250_000, monthly_cash_flow=60_000,
        is_greenfield=False, business_age_years=4, collateral_value=0, credit_score=710,
        owner_categories=["general"], location="urban", sells_to_large_buyers=True,
        goal="Stock up inventory ahead of the festive season and fund receivables"),
    "rural_food_processing": dict(
        business_name="Kisan Agro Foods", sector="manufacturing", purpose="setup",
        project_cost=1_500_000, owner_contribution=100_000, monthly_cash_flow=40_000,
        is_greenfield=True, business_age_years=0, collateral_value=0, credit_score=700,
        owner_categories=["sc_st"], location="rural",
        goal="Start a spice grinding and packaging unit in the village"),
    "small_tailoring_unit": dict(
        business_name="Silai Studio", sector="services", purpose="machinery",
        project_cost=45_000, owner_contribution=5_000, monthly_cash_flow=9_000,
        is_greenfield=False, business_age_years=1, collateral_value=0, credit_score=650,
        owner_categories=["general"], location="urban",
        goal="Buy two industrial sewing machines"),
    "overleveraged_cafe": dict(
        business_name="Brew Lane Cafe", sector="services", purpose="expansion",
        project_cost=3_000_000, owner_contribution=300_000, monthly_cash_flow=30_000,
        is_greenfield=False, business_age_years=3, collateral_value=0, credit_score=720,
        owner_categories=["general"], location="urban",
        goal="Open a second outlet with full kitchen fit-out"),
}
