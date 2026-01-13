"""
Trace data structure for AgentDiff.

A Trace represents a complete agent execution, containing multiple spans
that form an execution tree.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Dict, List
import uuid

from .span import Span, SpanStatus


@dataclass
class Trace:
    """
    Represents a complete agent execution.

    A trace contains multiple spans organized in a tree structure,
    along with aggregate metrics and metadata.
    """
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = "1.0.0"

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None

    # Spans
    spans: List[Span] = field(default_factory=list)
    root_span_id: Optional[str] = None

    # Aggregated metrics
    total_tokens: int = 0
    total_cost: float = 0.0
    total_duration_ms: float = 0.0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Git-like references for version control
    parent_trace_id: Optional[str] = None
    branch: str = "main"
    commit_message: Optional[str] = None

    # Status tracking
    status: str = "running"
    error: Optional[str] = None

    def finish(self):
        """Mark trace as finished and calculate aggregates."""
        self.end_time = datetime.utcnow()
        if self.start_time and self.end_time:
            self.total_duration_ms = (
                self.end_time - self.start_time
            ).total_seconds() * 1000

        # Calculate totals from spans
        self.total_tokens = 0
        self.total_cost = 0.0
        has_failure = False

        for span in self.spans:
            if span.token_usage:
                self.total_tokens += span.token_usage.total_tokens
                self.total_cost += span.token_usage.total_cost
            if span.status == SpanStatus.FAILED:
                has_failure = True

        self.status = "failed" if has_failure else "completed"

    def add_span(self, span: Span):
        """Add a span to the trace."""
        span.trace_id = self.trace_id
        self.spans.append(span)
        if not self.root_span_id and not span.parent_span_id:
            self.root_span_id = span.span_id

    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a span by ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None

    def get_root_span(self) -> Optional[Span]:
        """Get the root span of the trace."""
        if self.root_span_id:
            return self.get_span(self.root_span_id)
        return None

    def get_child_spans(self, parent_span_id: str) -> List[Span]:
        """Get all child spans of a parent span."""
        return [s for s in self.spans if s.parent_span_id == parent_span_id]

    def get_span_tree(self) -> Dict[str, Any]:
        """
        Build a tree representation of spans.

        Returns a nested dictionary with spans and their children.
        """
        def build_tree(span_id: Optional[str]) -> List[Dict[str, Any]]:
            children = []
            for span in self.spans:
                if span.parent_span_id == span_id:
                    node = {
                        "span": span,
                        "children": build_tree(span.span_id),
                    }
                    children.append(node)
            return children

        root = self.get_root_span()
        if root:
            return {
                "span": root,
                "children": build_tree(root.span_id),
            }
        return {"span": None, "children": build_tree(None)}

    def get_llm_calls(self) -> List[Span]:
        """Get all LLM call spans."""
        from .span import SpanType
        return [s for s in self.spans if s.span_type == SpanType.LLM_CALL]

    def get_tool_calls(self) -> List[Span]:
        """Get all tool call spans."""
        from .span import SpanType
        return [s for s in self.spans if s.span_type == SpanType.TOOL_CALL]

    def get_failed_spans(self) -> List[Span]:
        """Get all failed spans."""
        return [s for s in self.spans if s.status == SpanStatus.FAILED]

    def add_tag(self, tag: str):
        """Add a tag to the trace."""
        if tag not in self.tags:
            self.tags.append(tag)

    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "version": self.version,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "spans": [s.to_dict() for s in self.spans],
            "root_span_id": self.root_span_id,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "total_duration_ms": self.total_duration_ms,
            "metadata": self.metadata,
            "tags": self.tags,
            "parent_trace_id": self.parent_trace_id,
            "branch": self.branch,
            "commit_message": self.commit_message,
            "status": self.status,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trace":
        """Create trace from dictionary."""
        trace = cls(
            trace_id=data.get("trace_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            root_span_id=data.get("root_span_id"),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            total_duration_ms=data.get("total_duration_ms", 0.0),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            parent_trace_id=data.get("parent_trace_id"),
            branch=data.get("branch", "main"),
            commit_message=data.get("commit_message"),
            status=data.get("status", "running"),
            error=data.get("error"),
        )

        # Parse timestamps
        if data.get("start_time"):
            if isinstance(data["start_time"], str):
                trace.start_time = datetime.fromisoformat(data["start_time"])
            else:
                trace.start_time = data["start_time"]

        if data.get("end_time"):
            if isinstance(data["end_time"], str):
                trace.end_time = datetime.fromisoformat(data["end_time"])
            else:
                trace.end_time = data["end_time"]

        # Parse spans
        if data.get("spans"):
            trace.spans = [Span.from_dict(s) for s in data["spans"]]

        return trace

    def __repr__(self) -> str:
        return (
            f"Trace(id={self.trace_id[:8]}..., name={self.name!r}, "
            f"spans={len(self.spans)}, status={self.status})"
        )
