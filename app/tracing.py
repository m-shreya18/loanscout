"""Execution tracing.

Every request always gets a local trace (returned in the API response). When Langfuse keys
are configured the same structure is exported to Langfuse: one root `agent` observation per
request, a child observation per graph node (agent / retriever / tool / evaluator types),
LLM generations via the LangChain CallbackHandler, and numeric scores for retrieval
quality, DSCR, iterations and blocking issues.
"""
from __future__ import annotations

import contextlib
import logging
import time
from typing import Any

from app.config import get_settings

log = logging.getLogger("loanscout.tracing")
_client = None


def langfuse_client():
    global _client
    s = get_settings()
    if not s.langfuse_enabled:
        return None
    if _client is None:
        try:
            from langfuse import Langfuse

            _client = Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key,
                               host=s.langfuse_host)
        except Exception as e:  # never let tracing break the advisor
            log.warning("Langfuse disabled: %s", e)
            return None
    return _client


def _summarise(obj: Any, limit: int = 4000) -> Any:
    """Keep trace payloads small and JSON-friendly."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    text = repr(obj) if not isinstance(obj, (dict, list, str, int, float, type(None))) else obj
    if isinstance(text, str) and len(text) > limit:
        return text[:limit] + "…"
    return text


class RequestTracer:
    def __init__(self, request_id: str, name: str = "loanscout.plan"):
        self.request_id = request_id
        self.events: list[dict] = []
        self.lf = langfuse_client()
        self._root_cm = None
        self.root = None
        self.trace_id: str | None = None
        self.name = name

    # -------------------------------------------------------------- lifecycle
    def __enter__(self):
        if self.lf:
            try:
                self._root_cm = self.lf.start_as_current_observation(
                    name=self.name, as_type="agent", metadata={"request_id": self.request_id})
                self.root = self._root_cm.__enter__()
                self.trace_id = self.root.trace_id
            except Exception as e:
                log.warning("Langfuse root span failed: %s", e)
                self.lf = None
        return self

    def __exit__(self, *exc):
        if self._root_cm:
            with contextlib.suppress(Exception):
                self._root_cm.__exit__(*exc)
            with contextlib.suppress(Exception):
                self.lf.flush()

    def set_io(self, input: Any = None, output: Any = None):
        if self.root:
            with contextlib.suppress(Exception):
                self.root.update(input=_summarise(input), output=_summarise(output))

    @property
    def url(self) -> str | None:
        if self.lf and self.trace_id:
            with contextlib.suppress(Exception):
                return self.lf.get_trace_url(trace_id=self.trace_id)
        return None

    def callbacks(self) -> list:
        """LangChain callback handler so Gemini calls show up as generations."""
        if not self.lf:
            return []
        try:
            from langfuse.langchain import CallbackHandler

            return [CallbackHandler()]
        except Exception:
            return []

    # ----------------------------------------------------------------- spans
    @contextlib.contextmanager
    def span(self, name: str, as_type: str = "span", input: Any = None):
        rec = {"step": name, "type": as_type, "started_at": time.time()}
        lf_cm = obs = None
        if self.lf and self.root:
            try:
                lf_cm = self.lf.start_as_current_observation(
                    name=name, as_type=as_type, input=_summarise(input),
                    trace_context={"trace_id": self.trace_id, "parent_span_id": self.root.id})
                obs = lf_cm.__enter__()
            except Exception:
                lf_cm = obs = None
        holder: dict = {}
        try:
            yield holder
        finally:
            rec["duration_ms"] = round((time.time() - rec.pop("started_at")) * 1000, 2)
            rec.update({k: v for k, v in holder.items() if k != "output_full"})
            self.events.append(rec)
            if obs is not None:
                with contextlib.suppress(Exception):
                    obs.update(output=_summarise(holder.get("output_full", holder.get("output"))),
                               metadata={k: v for k, v in holder.items()
                                         if k not in ("output", "output_full")})
                with contextlib.suppress(Exception):
                    lf_cm.__exit__(None, None, None)

    def score(self, name: str, value: float, comment: str | None = None):
        self.events.append({"step": "score", "name": name, "value": value, "comment": comment})
        if self.lf and self.trace_id:
            with contextlib.suppress(Exception):
                self.lf.create_score(name=name, value=float(value), trace_id=self.trace_id,
                                     comment=comment)
