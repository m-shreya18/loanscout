"""Loads the scheme knowledge base and exposes structured fee-table lookups."""
from __future__ import annotations

import json
from functools import lru_cache

from app.config import get_settings


@lru_cache
def load_schemes() -> dict[str, dict]:
    with open(get_settings().knowledge_file, encoding="utf-8") as f:
        data = json.load(f)
    return {s["id"]: s for s in data["schemes"]}


def get_scheme(scheme_id: str) -> dict:
    schemes = load_schemes()
    if scheme_id not in schemes:
        raise KeyError(f"unknown scheme '{scheme_id}'")
    return schemes[scheme_id]


def fee_table(scheme_id: str) -> dict:
    """Structured fee-table lookup used by the planner and the skeptic."""
    s = get_scheme(scheme_id)
    f = s["fees"]
    return {
        "scheme_id": scheme_id,
        "interest_rate_pct": s["interest_rate"],
        "processing_fee_pct": f.get("processing_fee_pct", 0),
        "processing_fee_cap": f.get("processing_fee_cap", 0),
        "gst_on_fees_pct": f.get("gst_on_fees_pct", 0),
        "guarantee_fee_pct_annual": f.get("guarantee_fee_pct_annual", 0),
        "guarantee_fee_slabs": f.get("guarantee_fee_slabs"),
        "prepayment_penalty_pct": f.get("prepayment_penalty_pct", 0),
        "collateral_required": s["collateral_required"],
        "max_tenure_months": s["max_tenure_months"],
        "max_moratorium_months": s["max_moratorium_months"],
        "amount_range": [s["min_amount"], s["max_amount"]],
        "max_project_share": s.get("max_project_share", 1.0),
    }


def fee_table_text(scheme: dict) -> str:
    t = fee_table(scheme["id"])
    lines = [
        f"Fee table for {scheme['name']}:",
        f"interest rate {t['interest_rate_pct']['min']} to {t['interest_rate_pct']['max']} percent per annum;",
        f"processing fee {t['processing_fee_pct']} percent capped at Rs {t['processing_fee_cap']} plus GST {t['gst_on_fees_pct']} percent;",
        f"annual guarantee fee {t['guarantee_fee_pct_annual']} percent;",
        f"prepayment penalty {t['prepayment_penalty_pct']} percent;",
        f"collateral {'required' if t['collateral_required'] else 'not required'};",
        f"loan amount Rs {t['amount_range'][0]} to Rs {t['amount_range'][1]}; tenure up to {t['max_tenure_months']} months;",
        f"moratorium up to {t['max_moratorium_months']} months.",
    ]
    return " ".join(lines)
