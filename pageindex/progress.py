"""Live progress of ingest and delete, reported to an optional host sink"""
from contextvars import ContextVar
from typing import Protocol

current_doc_id: ContextVar[str | None] = ContextVar("hybrid_rag_doc_id", default=None)
current_workspace: ContextVar[str | None] = ContextVar("hybrid_rag_workspace", default=None)

class ProgressSink(Protocol):
    def stage(self, name: str) -> None:
        """A pipeline stage began (``parse``, ``chunk``, ``extract``, ``merge``, …)."""
    def skip(self, name: str, reason: str) -> None:
        """A stage was skipped on purpose."""
    def activity(self, message: str, level: str = "info") -> None:
        """Something worth showing inside the current stage (a wait, a warning)."""

_sink: ProgressSink | None = None

def set_sink(sink: ProgressSink | None) -> None:
    global _sink
    _sink = sink

def stage(name: str) -> None:
    if _sink:
        _sink.stage(name)

def skip(name: str, reason: str) -> None:
    if _sink:
        _sink.skip(name, reason)

def activity(message: str, level: str = "info") -> None:
    if _sink:
        _sink.activity(message, level)