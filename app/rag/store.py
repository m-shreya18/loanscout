"""Hierarchical RAG over ChromaDB.

Level 1  `scheme_summaries`  — one document per scheme. Metadata flags (sector_*, purpose_*)
                              let us pre-filter to schemes that fit the request ("scheme-aware").
Level 2  `scheme_sections`   — overview / eligibility / fees / fee_table chunks, queried only
                              inside the schemes chosen at level 1, one query per aspect.
"""
from __future__ import annotations

import re
from functools import lru_cache

import chromadb

from app.config import get_settings
from app.models import BusinessProfile, Citation
from app.rag.embeddings import Embedder
from app.rag.knowledge import fee_table_text, load_schemes

L1 = "scheme_summaries"
L2 = "scheme_sections"
SECTORS = ["manufacturing", "services", "trading"]
PURPOSES = ["setup", "machinery", "working_capital", "expansion"]


def _sentences(text: str, max_chars: int = 320) -> list[str]:
    parts = re.split(r"(?<=[.;])\s+", text)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) > max_chars and cur:
            chunks.append(cur.strip())
            cur = ""
        cur += " " + p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


class SchemeStore:
    def __init__(self, persist_dir: str | None = None):
        s = get_settings()
        self.client = chromadb.PersistentClient(path=persist_dir or s.chroma_dir)
        self.embedder = Embedder()
        meta = {"hnsw:space": "cosine"}
        self.l1 = self.client.get_or_create_collection(L1, metadata=meta, embedding_function=None)
        self.l2 = self.client.get_or_create_collection(L2, metadata=meta, embedding_function=None)
        if self.l1.count() == 0:
            self.ingest()

    # ---------------------------------------------------------------- ingest
    def ingest(self) -> dict:
        for name in (L1, L2):
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        meta = {"hnsw:space": "cosine"}
        self.l1 = self.client.get_or_create_collection(L1, metadata=meta, embedding_function=None)
        self.l2 = self.client.get_or_create_collection(L2, metadata=meta, embedding_function=None)

        ids1, docs1, metas1 = [], [], []
        ids2, docs2, metas2 = [], [], []
        for sid, s in load_schemes().items():
            el = s["eligibility"]
            m = {"scheme_id": sid, "provider_type": s["provider_type"], "instrument": s["instrument"]}
            m.update({f"sector_{x}": x in el["sectors"] for x in SECTORS})
            m.update({f"purpose_{x}": x in el["purposes"] for x in PURPOSES})
            ids1.append(sid)
            docs1.append(f"{s['name']}. {s['docs']['overview']} {s['docs']['eligibility']}")
            metas1.append(m)

            sections = dict(s["docs"])
            sections["fee_table"] = fee_table_text(s)
            for section, text in sections.items():
                for i, chunk in enumerate(_sentences(text)):
                    ids2.append(f"{sid}::{section}::{i}")
                    docs2.append(f"[{s['name']} — {section}] {chunk}")
                    metas2.append({"scheme_id": sid, "section": section})

        self.l1.add(ids=ids1, documents=docs1, metadatas=metas1,
                    embeddings=self.embedder.embed_documents(docs1))
        self.l2.add(ids=ids2, documents=docs2, metadatas=metas2,
                    embeddings=self.embedder.embed_documents(docs2))
        return {"schemes": len(ids1), "chunks": len(ids2)}

    # ------------------------------------------------------------- retrieval
    @staticmethod
    def build_queries(p: BusinessProfile) -> dict[str, str]:
        cats = " ".join(c for c in p.owner_categories if c != "general")
        stage = ("greenfield new unit subsidy margin money" if p.is_greenfield
                 else f"existing established business vintage {p.business_age_years} years")
        coll = "collateral available" if p.collateral_value > 0 else "no collateral collateral-free"
        base = f"{p.sector} {p.purpose.replace('_', ' ')} loan {stage} {cats} {p.location} {p.goal}"
        return {
            "discovery": f"{base} {coll} business finance scheme",
            "eligibility": f"eligibility {cats} {p.sector} {stage} credit score contribution",
            "fees": f"fee table interest rate processing fee guarantee fee {coll} tenure moratorium",
        }

    def retrieve(self, p: BusinessProfile, k1: int = 6, k2_per_aspect: int = 8) -> dict:
        q = self.build_queries(p)
        # Guarantees (CGTMSE) are not funding sources; they are pulled in at level 2 instead.
        where = {"$and": [{f"sector_{p.sector}": True}, {f"purpose_{p.purpose}": True},
                          {"instrument": {"$ne": "guarantee"}}]}
        r1 = self.l1.query(query_embeddings=[self.embedder.embed_query(q["discovery"])],
                           n_results=k1, where=where)
        top = [
            {"scheme_id": m["scheme_id"], "score": round(1 - d, 4)}
            for m, d in zip(r1["metadatas"][0], r1["distances"][0])
        ]
        top_ids = [t["scheme_id"] for t in top]
        # CGTMSE is a guarantee add-on; keep it reachable for the fee lookups.
        l2_ids = list(dict.fromkeys(top_ids + ["cgtmse"]))

        citations: list[Citation] = []
        for aspect in ("eligibility", "fees"):
            section_filter = {"section": {"$in": ["eligibility", "overview"]}} if aspect == "eligibility" \
                else {"section": {"$in": ["fees", "fee_table"]}}
            r2 = self.l2.query(
                query_embeddings=[self.embedder.embed_query(q[aspect])],
                n_results=min(k2_per_aspect * 2, self.l2.count()),
                where={"$and": [{"scheme_id": {"$in": l2_ids}}, section_filter]},
            )
            for doc, m, d in zip(r2["documents"][0], r2["metadatas"][0], r2["distances"][0]):
                citations.append(Citation(scheme_id=m["scheme_id"], section=m["section"],
                                          text=doc, score=round(1 - d, 4)))

        # Keep the best chunk per (scheme, section).
        best: dict[tuple, Citation] = {}
        for c in citations:
            key = (c.scheme_id, c.section)
            if key not in best or c.score > best[key].score:
                best[key] = c
        citations = sorted(best.values(), key=lambda c: -c.score)

        covered = {c.scheme_id for c in citations}
        metrics = {
            "queries": q,
            "level1_filter": where,
            "level1_hits": top,
            "level1_mean_score": round(sum(t["score"] for t in top) / len(top), 4) if top else 0.0,
            "level2_chunks": len(citations),
            "level2_mean_score": round(sum(c.score for c in citations) / len(citations), 4) if citations else 0.0,
            "section_coverage": round(len(covered & set(top_ids)) / len(top_ids), 3) if top_ids else 0.0,
        }
        return {"scheme_ids": top_ids, "citations": citations, "metrics": metrics}


@lru_cache
def get_store() -> SchemeStore:
    return SchemeStore()
