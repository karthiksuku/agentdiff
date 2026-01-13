"""
Span data structures for AgentDiff.

A Span represents a single operation in an agent trace, such as an LLM call,
tool invocation, or reasoning step.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Dict, List
from enum import Enum
import uuid


class SpanType(Enum):
    """Types of spans that can be captured."""
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    MEMORY_ACCESS = "memory_access"
    AGENT_STEP = "agent_step"
    PLANNING = "planning"
    REASONING = "reasoning"
    RETRIEVAL = "retrieval"
    EMBEDDING = "embedding"
    CUSTOM = "custom"


class SpanStatus(Enum):
    """Status of a span."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TokenUsage:
    """Token usage and cost information for an LLM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    # Cost in USD (calculated based on model pricing)
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    @property
    def cost(self) -> float:
        """Total cost for this token usage."""
        return self.total_cost

    def calculate_cost(self, input_price_per_1k: float, output_price_per_1k: float) -> float:
        """
        Calculate cost based on pricing.

        Args:
            input_price_per_1k: Price per 1000 input tokens
            output_price_per_1k: Price per 1000 output tokens

        Returns:
            Total cost in USD
        """
        self.input_cost = (self.input_tokens / 1000) * input_price_per_1k
        self.output_cost = (self.output_tokens / 1000) * output_price_per_1k
        self.total_cost = self.input_cost + self.output_cost
        return self.total_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenUsage":
        """Create from dictionary."""
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cached_tokens=data.get("cached_tokens", 0),
            input_cost=data.get("input_cost", 0.0),
            output_cost=data.get("output_cost", 0.0),
            total_cost=data.get("total_cost", 0.0),
        )


@dataclass
class Span:
    """
    Represents a single operation in an agent trace.

    Spans form a tree structure within a trace, with parent-child relationships
    representing nested operations.
    """
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    parent_span_id: Optional[str] = None

    name: str = ""
    span_type: SpanType = SpanType.CUSTOM
    status: SpanStatus = SpanStatus.RUNNING

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None

    # LLM specific
    model: Optional[str] = None
    provider: Optional[str] = None
    token_usage: Optional[TokenUsage] = None

    # Input/Output
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # For semantic analysis (embeddings)
    input_embedding: Optional[List[float]] = None
    output_embedding: Optional[List[float]] = None

    # Error tracking
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Agent-specific fields
    confidence_score: Optional[float] = None
    alternatives_considered: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: Optional[str] = None

    def finish(self, status: SpanStatus = SpanStatus.COMPLETED):
        """Mark span as finished."""
        self.end_time = datetime.utcnow()
        self.status = status
        if self.start_time and self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

    def fail(self, error: str, error_type: Optional[str] = None):
        """Mark span as failed with error."""
        self.error = error
        self.error_type = error_type
        self.finish(SpanStatus.FAILED)

    def add_tag(self, tag: str):
        """Add a tag to the span."""
        if tag not in self.tags:
            self.tags.append(tag)

    def set_input(self, data: Dict[str, Any]):
        """Set input data."""
        self.input_data = data

    def set_output(self, data: Dict[str, Any]):
        """Set output data."""
        self.output_data = data

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "span_type": self.span_type.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "model": self.model,
            "provider": self.provider,
            "token_usage": self.token_usage.to_dict() if self.token_usage else None,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "metadata": self.metadata,
            "tags": self.tags,
            "input_embedding": self.input_embedding,
            "output_embedding": self.output_embedding,
            "error": self.error,
            "error_type": self.error_type,
            "confidence_score": self.confidence_score,
            "alternatives_considered": self.alternatives_considered,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Span":
        """Create span from dictionary."""
        span = cls(
            span_id=data.get("span_id", str(uuid.uuid4())),
            trace_id=data.get("trace_id", ""),
            parent_span_id=data.get("parent_span_id"),
            name=data.get("name", ""),
            span_type=SpanType(data.get("span_type", "custom")),
            status=SpanStatus(data.get("status", "running")),
            model=data.get("model"),
            provider=data.get("provider"),
            input_data=data.get("input_data", {}),
            output_data=data.get("output_data", {}),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            input_embedding=data.get("input_embedding"),
            output_embedding=data.get("output_embedding"),
            error=data.get("error"),
            error_type=data.get("error_type"),
            confidence_score=data.get("confidence_score"),
            alternatives_considered=data.get("alternatives_considered", []),
            reasoning=data.get("reasoning"),
        )

        # Parse timestamps
        if data.get("start_time"):
            if isinstance(data["start_time"], str):
                span.start_time = datetime.fromisoformat(data["start_time"])
            else:
                span.start_time = data["start_time"]

        if data.get("end_time"):
            if isinstance(data["end_time"], str):
                span.end_time = datetime.fromisoformat(data["end_time"])
            else:
                span.end_time = data["end_time"]

        span.duration_ms = data.get("duration_ms")

        # Parse token usage
        if data.get("token_usage"):
            span.token_usage = TokenUsage.from_dict(data["token_usage"])

        return span

    def __repr__(self) -> str:
        return (
            f"Span(id={self.span_id[:8]}..., name={self.name!r}, "
            f"type={self.span_type.value}, status={self.status.value})"
        )
