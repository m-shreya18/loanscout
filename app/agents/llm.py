"""Optional Gemini LLM. Every agent works without it (deterministic mode); with a key the
planner drafts plans, the skeptic adds qualitative critique and the writer narrates."""
from __future__ import annotations

import json
import re
from functools import lru_cache

from app.config import get_settings


@lru_cache
def get_llm():
    s = get_settings()
    if not s.llm_enabled:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=s.llm_model, temperature=s.llm_temperature, google_api_key=s.google_api_key
    )


def call_llm(system: str, user: str, config: dict | None = None) -> str | None:
    llm = get_llm()
    if llm is None:
        return None
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        msg = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)], config=config)
        content = msg.content
        if isinstance(content, list):  # Gemini can return content parts
            content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
        return content
    except Exception as e:  # network / quota errors fall back to deterministic logic
        return f"__LLM_ERROR__ {e}"


def extract_json(text: str | None):
    if not text or text.startswith("__LLM_ERROR__"):
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    raw = m.group(1) if m else text
    start = min([i for i in (raw.find("{"), raw.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    try:
        return json.loads(raw[start:])
    except json.JSONDecodeError:
        end = max(raw.rfind("}"), raw.rfind("]"))
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None
