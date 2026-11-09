# LoanScout: Agentic Fintech Lending Advisor

A self-correcting multi-agent lending advisor that recommends the best way to fund a small-business project in India. It mixes owner equity, government schemes (Mudra, Stand-Up India, PMEGP), subsidies, CGTMSE guarantees, bank loans, NBFC credit and invoice discounting, then checks every rupee before it recommends anything.

Built with **Python, LangGraph, FastAPI, ChromaDB and Langfuse**, with optional **Gemini** reasoning.

```
retrieve ─► eligibility ─► planner ─► verify_costs ─► skeptic ──┐
 (RAG)        (rules)        ▲          (tool)                   │ blocking issues
                             └───────────────────────────────────┘ and progress possible
                                                                 │ otherwise
                                                                 ▼
                                                              finalize
```

## What it does

You give it a business profile: sector, purpose, project cost, own money, monthly cash flow, collateral, credit score, owner category and location. You get back a complete funding plan with:

- the funding mix (equity, subsidy and each loan with amount, rate, tenure, EMI, CGTMSE cover)
- the **verified** total cost of credit: interest, processing fee + GST, and the annual guarantee fee on the outstanding balance
- the **effective APR**, from an IRR on the actual cash flows
- the DSCR against a 1.25 target
- the skeptic's round-by-round critique and the open issues
- the eligibility reasons for every rejected scheme, plus the retrieved evidence with similarity scores
- retrieval-quality metrics and the full execution trace (or a Langfuse link)

## How each resume claim maps to code

| Claim | Where |
|---|---|
| Custom multi-agent pipeline (Python, LangGraph, FastAPI) | `app/graph.py` (StateGraph with a conditional loop), `app/agents/*`, `app/main.py` |
| Hierarchical RAG on ChromaDB, scheme-aware retrieval | `app/rag/store.py`. **Level 1** searches scheme summaries, with metadata pre-filters on sector, purpose and instrument. **Level 2** searches section chunks (overview, eligibility, fees, fee table) only inside the schemes level 1 picked, with one query per aspect. |
| Fee-table lookups | `app/rag/knowledge.py::fee_table()` gives structured lookups. Fee tables are also indexed as level-2 chunks. |
| Self-correction loop with a skeptic agent | `app/agents/skeptic.py::critique()` raises typed issues. `app/agents/planner.py::update_memory()` turns them into plan changes: attach CGTMSE, stretch tenure, re-rank by EMI, exclude a scheme. The loop stops when the plan is accepted, when no further progress is possible (it escalates), or at `MAX_ITERATIONS`. |
| Verified cost checks | `app/tools/finance.py` covers EMI, amortisation with moratorium, fee caps + GST, guarantee-fee slabs, and IRR-based APR. The LLM never does arithmetic. Any claimed cost more than 1% off is flagged and replaced. |
| Langfuse tracing of reasoning, tool calls and retrieval quality | `app/tracing.py`. Each request gets one root `agent` observation. Every node becomes a typed child observation (`retriever`, `agent`, `tool`, `evaluator`). Gemini calls are captured through the LangChain `CallbackHandler`. Scores recorded: `retrieval_eligible_recall`, `retrieval_level2_mean_similarity`, `final_dscr`, `self_correction_iterations`, `open_blocking_issues`. |

## Agents

| Node | Type | Job |
|---|---|---|
| `hierarchical_rag` | retriever | Two-level ChromaDB retrieval plus evidence citations |
| `eligibility_agent` | agent | Rule checks for every scheme (sector, purpose, greenfield, vintage, category, credit score, margin, ticket size, PMEGP caps). It also measures retrieval recall against these rules. |
| `planner_agent` | agent | Drafts and revises the funding mix. Deterministic greedy optimiser by default; Gemini JSON planner when a key is set, with schema validation and fallback. |
| `cost_verifier_tool` | tool | Recomputes the cost of every line item |
| `skeptic_agent` | evaluator | Checks for ineligible schemes, duplicate Mudra loans, amount, rate and tenure limits, collateral shortfall, funding gap or over-funding, low DSCR and cost mismatches (blocking). Warns about expensive NBFC concentration, prepayment penalties and TReDS receivables limits. With Gemini, it also adds qualitative advisories. |
| `report_writer` | agent | Sets the status and writes the summary. Template by default; a Gemini rewrite keeps every number unchanged. |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional: add GOOGLE_API_KEY / Langfuse keys

python -m scripts.demo          # run all six sample scenarios in the terminal
uvicorn app.main:app --reload   # web UI at http://localhost:8000, API docs at /docs
pytest -q                       # 16 tests
```

Docker: `docker compose up --build`

**No API key needed.** Without `GOOGLE_API_KEY` the same graph runs in deterministic mode, fully offline. The default `hash` embeddings need no model download. Add the key to switch the planner, skeptic advisories and summary writer to Gemini (`gemini-2.5-flash`). Add Langfuse keys to export traces.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/plan` | Full advisor run → `PlanResponse` |
| `POST` | `/retrieve` | Run only the hierarchical RAG step (useful for debugging) |
| `GET` | `/schemes` | All schemes with fee tables and eligibility rules |
| `GET` | `/schemes/{id}/fees` | Fee-table lookup |
| `GET` | `/scenarios` | Sample profiles |
| `POST` | `/ingest` | Re-index the knowledge base after editing `schemes.json` |
| `GET` | `/health` | Mode, Langfuse status and index size |

```bash
curl -X POST localhost:8000/plan -H 'Content-Type: application/json' -d '{
  "business_name": "Shakti Precision Works", "sector": "manufacturing", "purpose": "machinery",
  "project_cost": 4000000, "owner_contribution": 900000, "monthly_cash_flow": 95000,
  "is_greenfield": false, "business_age_years": 6, "collateral_value": 1500000,
  "credit_score": 760, "owner_categories": ["general"], "goal": "Two CNC machines"}'
```

## Sample results (deterministic mode)

```
existing_cnc_expansion  →  APPROVED_WITH_WARNINGS  (iterations=2)
  iter 1: DSCR=0.944  verdict=revise
      ✗ COLLATERAL_SHORTFALL: Bank MSME Term Loan needs collateral for Rs 3,100,000 but only Rs 1,500,000 is unpledged.
      ✗ DSCR_LOW: DSCR 0.94 is below 1.25 ... stretching tenures should fix this.
  iter 2: DSCR=2.282  verdict=accept
  Bank MSME Term Loan  Rs 3,100,000  10.39%  120m  EMI 41,639  cost 2,041,348  APR 11.90%  +CGTMSE

woman_bakery_greenfield →  APPROVED  (iterations=1)
  Own Rs 300,000 + PMEGP subsidy Rs 625,000 + PMEGP loan Rs 1,575,000 @10.7%  DSCR 1.46

overleveraged_cafe      →  NEEDS_ATTENTION  (iterations=3)
  ✗ DSCR_LOW: ... business can sustain about Rs 932,128 of term debt; roughly
    Rs 1,767,872 must come from extra equity or a smaller project.
```

The last case is deliberate. When no compliant plan exists, the skeptic escalates with the size of the shortfall instead of letting a bad plan through.

## Tests

- `test_finance.py` checks EMI against a textbook value, amortisation with moratorium, fee caps + GST, CGTMSE fees, the IRR APR and the inverse EMI.
- `test_rag.py` checks level-1 metadata filtering, that level 2 stays inside the chosen schemes, and that **every rule-eligible scheme is retrieved** in every scenario.
- `test_agents.py` checks eligibility rules, the subsidy path, the collateral and DSCR self-correction, escalation, and trace structure. It also has a **mocked-Gemini test**: round 1 invents a 7% rate, over-funds the project and under-reports cost; the skeptic rejects it, and round 2 is accepted.
- `test_api.py` runs FastAPI end to end, including 404 and 422 validation.

## Project layout

```
app/
  main.py               FastAPI app + static UI
  graph.py              LangGraph state machine, run_advisor()
  config.py, models.py  settings and Pydantic schemas
  tracing.py            local trace + Langfuse export
  agents/               eligibility, planner, skeptic, llm
  rag/                  embeddings, knowledge (fee tables), store (hierarchical Chroma)
  tools/finance.py      verified cost-of-credit maths
  knowledge/schemes.json  10 schemes with docs and fee tables
  static/index.html     single-page UI
scripts/                demo + sample scenarios
tests/                  pytest suite
```

## Disclaimer

The scheme figures in `app/knowledge/schemes.json` are illustrative, compiled for a portfolio project. Rates, fees and limits change and vary by lender. This is not financial advice. Verify with the lender or the official scheme portal before acting.
