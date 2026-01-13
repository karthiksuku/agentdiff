"""
AgentDiff - Visual Diff & Replay Debugger for AI Agents

A tool for capturing, comparing, and replaying AI agent executions.
"""

__version__ = "0.1.0"

from .core.tracer import (
    AgentDiffTracer,
    configure,
    get_tracer,
    trace,
    get_current_trace,
    get_current_span,
)
from .core.span import Span, SpanType, SpanStatus, TokenUsage
from .core.trace import Trace

__all__ = [
    # Version
    "__version__",
    # Core tracer
    "AgentDiffTracer",
    "configure",
    "get_tracer",
    "trace",
    "get_current_trace",
    "get_current_span",
    # Data models
    "Span",
    "SpanType",
    "SpanStatus",
    "TokenUsage",
    "Trace",
]
