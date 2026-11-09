"""FastAPI service for LoanScout."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.graph import run_advisor
from app.models import BusinessProfile, PlanResponse
from app.rag.knowledge import fee_table, load_schemes
from app.rag.store import get_store
from scripts.scenarios import SCENARIOS

logging.basicConfig(level=logging.INFO)
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    get_store()  # builds the Chroma index on first boot
    yield


app = FastAPI(
    lifespan=lifespan,
    title="LoanScout",
    description="Self-correcting multi-agent lending advisor for Indian MSME project funding.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "mode": "llm" if s.llm_enabled else "deterministic",
            "llm_model": s.llm_model if s.llm_enabled else None,
            "langfuse": s.langfuse_enabled, "schemes": len(load_schemes()),
            "indexed_chunks": get_store().l2.count()}


@app.get("/schemes")
def schemes():
    return [{"id": sid, "name": s["name"], "provider_type": s["provider_type"],
             "instrument": s["instrument"], "fee_table": fee_table(sid),
             "eligibility": s["eligibility"]} for sid, s in load_schemes().items()]


@app.get("/schemes/{scheme_id}/fees")
def scheme_fees(scheme_id: str):
    try:
        return fee_table(scheme_id)
    except KeyError as e:
        raise HTTPException(404, str(e)) from e


@app.get("/scenarios")
def scenarios():
    return SCENARIOS


@app.post("/retrieve")
def retrieve(profile: BusinessProfile):
    r = get_store().retrieve(profile)
    return {"scheme_ids": r["scheme_ids"], "metrics": r["metrics"],
            "citations": [c.model_dump() for c in r["citations"]]}


@app.post("/plan", response_model=PlanResponse)
def plan(profile: BusinessProfile):
    return run_advisor(profile)


@app.post("/ingest")
def ingest():
    load_schemes.cache_clear()
    return get_store().ingest()
