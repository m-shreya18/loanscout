"""Embedding backends.

`hash` (default) is a fully offline feature-hashing embedder over unigrams + bigrams with
a small finance synonym map. It needs no model download, is deterministic, and is good
enough for a ~40-chunk domain corpus. `gemini` uses Google's text-embedding model when a
GOOGLE_API_KEY is configured.
"""
from __future__ import annotations

import hashlib
import math
import re

from app.config import get_settings

DIM = 768
_STOP = set(
    "a an the of for to and or in on at by is are be with as from that this it its up per "
    "can will may not no into than then also about".split()
)
_SYNONYMS = {
    "women": "woman", "female": "woman", "lady": "woman",
    "sc": "sc_st", "st": "sc_st", "scheduled": "sc_st", "dalit": "sc_st",
    "new": "greenfield", "startup": "greenfield", "first-time": "greenfield",
    "charges": "fee", "fees": "fee", "cost": "fee", "costs": "fee",
    "security": "collateral", "mortgage": "collateral",
    "subsidy": "subsidy", "grant": "subsidy",
    "machine": "machinery", "machines": "machinery", "equipment": "machinery",
    "inventory": "working_capital", "stock": "working_capital", "receivables": "working_capital",
    "invoices": "invoice", "manufacturer": "manufacturing", "factory": "manufacturing",
    "village": "rural", "loans": "loan", "rates": "rate", "interest": "rate",
}


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_\-]+", text.lower().replace("working capital", "working_capital"))
    out = []
    for w in words:
        if w in _STOP:
            continue
        w = _SYNONYMS.get(w, w)
        if len(w) > 4 and w.endswith("s"):
            w = w[:-1]
        out.append(w)
    return out


def _bucket(feature: str) -> tuple[int, float]:
    h = hashlib.md5(feature.encode()).digest()
    idx = int.from_bytes(h[:4], "little") % DIM
    sign = 1.0 if h[4] & 1 else -1.0
    return idx, sign


def hash_embed(text: str) -> list[float]:
    toks = _tokens(text)
    feats: dict[str, float] = {}
    for t in toks:
        feats[t] = feats.get(t, 0) + 1.0
    for a, b in zip(toks, toks[1:]):
        k = f"{a}__{b}"
        feats[k] = feats.get(k, 0) + 0.5
    vec = [0.0] * DIM
    for f, c in feats.items():
        i, s = _bucket(f)
        vec[i] += s * (1 + math.log(c))
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class Embedder:
    def __init__(self):
        s = get_settings()
        self.backend = s.embedding_backend
        self._gemini = None
        if self.backend == "gemini":
            if not s.google_api_key:
                self.backend = "hash"
            else:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings

                self._gemini = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004", google_api_key=s.google_api_key
                )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._gemini:
            return self._gemini.embed_documents(texts)
        return [hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if self._gemini:
            return self._gemini.embed_query(text)
        return hash_embed(text)
