"""
Main tracing functionality for AgentDiff.

Provides decorators and context managers for capturing agent executions.
"""

import functools
import asyncio
from typing import Optional, Dict, Any, Callable, TypeVar, Union
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime
import threading
import logging

from .span import Span, SpanType, SpanStatus, TokenUsage
from .trace import Trace
from .cost_tracker import get_cost_tracker
from ..storage.base import BaseStorage

logger = logging.getLogger(__name__)

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Thread-local storage for current trace context
_trace_context = threading.local()


def get_current_trace() -> Optional[Trace]:
    """Get the current trace from context."""
    return getattr(_trace_context, "trace", None)


def get_current_span() -> Optional[Span]:
    """Get the current span from context."""
    return getattr(_trace_context, "span", None)


def _set_current_trace(trace: Optional[Trace]) -> None:
    """Set the current trace in context."""
    _trace_context.trace = trace


def _set_current_span(span: Optional[Span]) -> None:
    """Set the current span in context."""
    _trace_context.span = span


class AgentDiffTracer:
    """
    Main tracer class for capturing agent executions.

    Usage:
        tracer = AgentDiffTracer(storage=SQLiteStore())

        @tracer.trace(name="my-agent")
        def my_agent(query):
            ...

        # Or as context manager
        with tracer.trace_context(name="my-agent") as trace:
            ...
    """

    def __init__(
        self,
        storage: BaseStorage,
        auto_instrument: bool = True,
        capture_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
        auto_calculate_cost: bool = True,
    ):
        """
        Initialize the tracer.

        Args:
            storage: Storage backend for persisting traces
            auto_instrument: Whether to auto-instrument LLM calls
            capture_embeddings: Whether to capture embeddings for semantic analysis
            embedding_model: Sentence transformer model for embeddings
            auto_calculate_cost: Whether to automatically calculate costs
        """
        self.storage = storage
        self.auto_instrument = auto_instrument
        self.capture_embeddings = capture_embeddings
        self.embedding_model = embedding_model
        self.auto_calculate_cost = auto_calculate_cost
        self._embedding_fn: Optional[Callable[[str], list]] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the tracer and storage."""
        if not self._initialized:
            await self.storage.initialize()
            if self.capture_embeddings:
                self._init_embeddings()
            self._initialized = True

    def _init_embeddings(self) -> None:
        """Initialize embedding model for semantic analysis."""
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.embedding_model)
            self._embedding_fn = lambda text: model.encode(text).tolist()
            logger.info(f"Initialized embedding model: {self.embedding_model}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Semantic diff disabled. "
                "Install with: pip install sentence-transformers"
            )
            self.capture_embeddings = False

    def _compute_embedding(self, text: str) -> Optional[list]:
        """Compute embedding for text."""
        if not self._embedding_fn or not text:
            return None
        try:
            # Truncate to avoid very long embeddings
            truncated = text[:512]
            return self._embedding_fn(truncated)
        except Exception as e:
            logger.warning(f"Failed to compute embedding: {e}")
            return None

    def trace(
        self,
        name: str,
        version: str = "1.0.0",
        branch: str = "main",
        tags: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> Callable[[F], F]:
        """
        Decorator to trace a function.

        Args:
            name: Name for the trace
            version: Version string
            branch: Branch name (git-like)
            tags: Optional list of tags
            metadata: Optional metadata dictionary

        Returns:
            Decorated function
        """

        def decorator(func: F) -> F:
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                async with self.trace_context_async(
                    name=name,
                    version=version,
                    branch=branch,
                    tags=tags or [],
                    metadata=metadata or {},
                ):
                    return await func(*args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.trace_context(
                    name=name,
                    version=version,
                    branch=branch,
                    tags=tags or [],
                    metadata=metadata or {},
                ):
                    return func(*args, **kwargs)

            if asyncio.iscoroutinefunction(func):
                return async_wrapper  # type: ignore
            return sync_wrapper  # type: ignore

        return decorator

    @contextmanager
    def trace_context(
        self,
        name: str,
        version: str = "1.0.0",
        branch: str = "main",
        tags: Optional[list] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Context manager for tracing (synchronous).

        Args:
            name: Name for the trace
            version: Version string
            branch: Branch name
            tags: Optional list of tags
            metadata: Optional metadata dictionary

        Yields:
            The Trace object
        """
        trace = Trace(
            name=name,
            version=version,
            branch=branch,
            tags=tags or [],
            metadata=metadata or {},
        )

        _set_current_trace(trace)
        _set_current_span(None)

        try:
            yield trace
        finally:
            trace.finish()

            # Save trace synchronously
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule the save for later
                    asyncio.create_task(self.storage.save_trace(trace))
                else:
                    loop.run_until_complete(self.storage.save_trace(trace))
            except RuntimeError:
                # No event loop, create one
                asyncio.run(self.storage.save_trace(trace))

            _set_current_trace(None)
            _set_current_span(None)

    @asynccontextmanager
    async def trace_context_async(
        self,
        name: str,
        version: str = "1.0.0",
        branch: str = "main",
        tags: Optional[list] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Async context manager for tracing.

        Args:
            name: Name for the trace
            version: Version string
            branch: Branch name
            tags: Optional list of tags
            metadata: Optional metadata dictionary

        Yields:
            The Trace object
        """
        trace = Trace(
            name=name,
            version=version,
            branch=branch,
            tags=tags or [],
            metadata=metadata or {},
        )

        _set_current_trace(trace)
        _set_current_span(None)

        try:
            yield trace
        finally:
            trace.finish()
            await self.storage.save_trace(trace)
            _set_current_trace(None)
            _set_current_span(None)

    @contextmanager
    def span(
        self,
        name: str,
        span_type: SpanType = SpanType.CUSTOM,
        metadata: Optional[dict] = None,
    ):
        """
        Create a span within a trace.

        Args:
            name: Name for the span
            span_type: Type of span
            metadata: Optional metadata dictionary

        Yields:
            The Span object

        Raises:
            RuntimeError: If no active trace exists
        """
        trace = get_current_trace()
        if not trace:
            raise RuntimeError("No active trace. Use @trace or trace_context first.")

        parent_span = get_current_span()

        span = Span(
            trace_id=trace.trace_id,
            parent_span_id=parent_span.span_id if parent_span else None,
            name=name,
            span_type=span_type,
            metadata=metadata or {},
        )

        _set_current_span(span)
        trace.add_span(span)

        try:
            yield span
            span.status = SpanStatus.COMPLETED
        except Exception as e:
            span.status = SpanStatus.FAILED
            span.error = str(e)
            span.error_type = type(e).__name__
            raise
        finally:
            span.end_time = datetime.utcnow()
            if span.start_time:
                span.duration_ms = (
                    span.end_time - span.start_time
                ).total_seconds() * 1000

            # Generate embeddings if enabled
            if self.capture_embeddings:
                if span.input_data:
                    input_text = str(span.input_data)
                    span.input_embedding = self._compute_embedding(input_text)
                if span.output_data:
                    output_text = str(span.output_data)
                    span.output_embedding = self._compute_embedding(output_text)

            _set_current_span(parent_span)

    def log_llm_call(
        self,
        model: str,
        provider: str,
        input_messages: list,
        output_message: str,
        token_usage: Dict[str, int],
        **kwargs: Any,
    ) -> None:
        """
        Log an LLM call within the current span.

        Args:
            model: Model identifier
            provider: Provider name (openai, anthropic, etc.)
            input_messages: Input messages list
            output_message: Output/response message
            token_usage: Token usage dictionary
            **kwargs: Additional metadata
        """
        span = get_current_span()
        if not span:
            logger.warning("No active span for LLM call logging")
            return

        span.model = model
        span.provider = provider
        span.span_type = SpanType.LLM_CALL
        span.input_data = {"messages": input_messages}
        span.output_data = {"response": output_message}

        # Create token usage
        usage = TokenUsage(
            input_tokens=token_usage.get("prompt_tokens", 0),
            output_tokens=token_usage.get("completion_tokens", 0),
            total_tokens=token_usage.get("total_tokens", 0),
            cached_tokens=token_usage.get("cached_tokens", 0),
        )

        # Calculate cost if enabled
        if self.auto_calculate_cost:
            cost_tracker = get_cost_tracker()
            usage.calculate_cost(
                cost_tracker.get_pricing(model).input_price_per_1k
                if cost_tracker.get_pricing(model)
                else 0,
                cost_tracker.get_pricing(model).output_price_per_1k
                if cost_tracker.get_pricing(model)
                else 0,
            )

        span.token_usage = usage
        span.metadata.update(kwargs)

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        **kwargs: Any,
    ) -> None:
        """
        Log a tool call within the current span.

        Args:
            tool_name: Name of the tool
            tool_input: Tool input parameters
            tool_output: Tool output/result
            **kwargs: Additional metadata
        """
        span = get_current_span()
        if not span:
            logger.warning("No active span for tool call logging")
            return

        span.span_type = SpanType.TOOL_CALL
        span.name = f"tool:{tool_name}"
        span.input_data = {"tool": tool_name, "input": tool_input}
        span.output_data = {"output": tool_output}
        span.metadata.update(kwargs)

    def log_decision(
        self,
        decision: str,
        confidence: float,
        alternatives: Optional[list] = None,
        reasoning: Optional[str] = None,
    ) -> None:
        """
        Log an agent decision point.

        Args:
            decision: The decision made
            confidence: Confidence score (0-1)
            alternatives: Alternative options considered
            reasoning: Reasoning for the decision
        """
        span = get_current_span()
        if not span:
            logger.warning("No active span for decision logging")
            return

        span.span_type = SpanType.REASONING
        span.confidence_score = confidence
        span.alternatives_considered = alternatives or []
        span.reasoning = reasoning
        span.output_data["decision"] = decision

    def log_retrieval(
        self,
        query: str,
        results: list,
        source: str = "unknown",
        **kwargs: Any,
    ) -> None:
        """
        Log a retrieval operation (RAG).

        Args:
            query: Search query
            results: Retrieved results
            source: Source of retrieval
            **kwargs: Additional metadata
        """
        span = get_current_span()
        if not span:
            logger.warning("No active span for retrieval logging")
            return

        span.span_type = SpanType.RETRIEVAL
        span.input_data = {"query": query, "source": source}
        span.output_data = {"results": results, "count": len(results)}
        span.metadata.update(kwargs)

    def log_memory(
        self,
        operation: str,
        key: Optional[str] = None,
        value: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        Log a memory access operation.

        Args:
            operation: Operation type (read, write, delete)
            key: Memory key
            value: Memory value
            **kwargs: Additional metadata
        """
        span = get_current_span()
        if not span:
            logger.warning("No active span for memory logging")
            return

        span.span_type = SpanType.MEMORY_ACCESS
        span.input_data = {"operation": operation, "key": key}
        span.output_data = {"value": value}
        span.metadata.update(kwargs)


# Global tracer instance (configured by user)
_global_tracer: Optional[AgentDiffTracer] = None


def configure(storage: BaseStorage, **kwargs: Any) -> AgentDiffTracer:
    """
    Configure the global tracer.

    Args:
        storage: Storage backend
        **kwargs: Additional tracer options

    Returns:
        The configured AgentDiffTracer
    """
    global _global_tracer
    _global_tracer = AgentDiffTracer(storage=storage, **kwargs)
    return _global_tracer


def get_tracer() -> AgentDiffTracer:
    """
    Get the global tracer.

    Returns:
        The global AgentDiffTracer

    Raises:
        RuntimeError: If tracer is not configured
    """
    if not _global_tracer:
        raise RuntimeError("AgentDiff not configured. Call agentdiff.configure() first.")
    return _global_tracer


def trace(name: str, **kwargs: Any) -> Callable[[F], F]:
    """
    Decorator using global tracer.

    Args:
        name: Name for the trace
        **kwargs: Additional trace options

    Returns:
        Decorator function
    """
    return get_tracer().trace(name=name, **kwargs)


def span(
    name: str,
    span_type: SpanType = SpanType.CUSTOM,
    metadata: Optional[dict] = None,
):
    """
    Context manager for creating a span using global tracer.

    Args:
        name: Name for the span
        span_type: Type of span
        metadata: Optional metadata

    Returns:
        Context manager yielding the Span
    """
    return get_tracer().span(name=name, span_type=span_type, metadata=metadata)
