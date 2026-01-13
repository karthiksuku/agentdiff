"""
Generic LLM hook for manual tracing of any LLM provider.
"""

from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from ..core.tracer import get_tracer, get_current_span, get_current_trace
from ..core.span import Span, SpanType, SpanStatus, TokenUsage
from ..core.cost_tracker import get_cost_tracker


class GenericLLMHook:
    """
    Generic hook for manually tracing LLM calls from any provider.

    Usage:
        from agentdiff.capture import GenericLLMHook

        hook = GenericLLMHook()

        # Option 1: Context manager
        with hook.llm_call("gpt-4", "openai") as span:
            span.set_input({"messages": [...]})
            response = my_llm_call(...)
            span.set_output({"response": response})

        # Option 2: Manual logging
        hook.log_llm_call(
            model="gpt-4",
            provider="openai",
            messages=[...],
            response="...",
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
        )
    """

    def __init__(self, auto_calculate_cost: bool = True):
        """
        Initialize the hook.

        Args:
            auto_calculate_cost: Whether to auto-calculate costs
        """
        self.auto_calculate_cost = auto_calculate_cost

    @contextmanager
    def llm_call(
        self,
        model: str,
        provider: str = "unknown",
        name: Optional[str] = None,
    ):
        """
        Context manager for tracing an LLM call.

        Args:
            model: Model identifier
            provider: Provider name
            name: Optional span name

        Yields:
            Span object for the LLM call
        """
        trace = get_current_trace()
        parent_span = get_current_span()

        span = Span(
            trace_id=trace.trace_id if trace else "",
            parent_span_id=parent_span.span_id if parent_span else None,
            name=name or f"llm:{model}",
            span_type=SpanType.LLM_CALL,
            model=model,
            provider=provider,
        )

        if trace:
            trace.add_span(span)

        try:
            yield span
            span.finish(SpanStatus.COMPLETED)
        except Exception as e:
            span.error = str(e)
            span.error_type = type(e).__name__
            span.finish(SpanStatus.FAILED)
            raise

    @contextmanager
    def tool_call(
        self,
        tool_name: str,
        name: Optional[str] = None,
    ):
        """
        Context manager for tracing a tool call.

        Args:
            tool_name: Name of the tool
            name: Optional span name

        Yields:
            Span object for the tool call
        """
        trace = get_current_trace()
        parent_span = get_current_span()

        span = Span(
            trace_id=trace.trace_id if trace else "",
            parent_span_id=parent_span.span_id if parent_span else None,
            name=name or f"tool:{tool_name}",
            span_type=SpanType.TOOL_CALL,
        )

        span.input_data["tool"] = tool_name

        if trace:
            trace.add_span(span)

        try:
            yield span
            span.finish(SpanStatus.COMPLETED)
        except Exception as e:
            span.error = str(e)
            span.error_type = type(e).__name__
            span.finish(SpanStatus.FAILED)
            raise

    def log_llm_call(
        self,
        model: str,
        provider: str,
        messages: List[Dict[str, Any]],
        response: str,
        token_usage: Optional[Dict[str, int]] = None,
        **metadata,
    ) -> Span:
        """
        Log an LLM call without using a context manager.

        Args:
            model: Model identifier
            provider: Provider name
            messages: Input messages
            response: Response text
            token_usage: Token usage dict
            **metadata: Additional metadata

        Returns:
            Created Span
        """
        trace = get_current_trace()
        parent_span = get_current_span()

        span = Span(
            trace_id=trace.trace_id if trace else "",
            parent_span_id=parent_span.span_id if parent_span else None,
            name=f"llm:{model}",
            span_type=SpanType.LLM_CALL,
            model=model,
            provider=provider,
            input_data={"messages": messages},
            output_data={"response": response},
            metadata=metadata,
        )

        # Handle token usage
        if token_usage:
            span.token_usage = TokenUsage(
                input_tokens=token_usage.get("prompt_tokens", 0),
                output_tokens=token_usage.get("completion_tokens", 0),
                total_tokens=token_usage.get("total_tokens", 0),
            )

            if self.auto_calculate_cost:
                cost_tracker = get_cost_tracker()
                pricing = cost_tracker.get_pricing(model)
                if pricing:
                    span.token_usage.calculate_cost(
                        pricing.input_price_per_1k,
                        pricing.output_price_per_1k,
                    )

        span.finish(SpanStatus.COMPLETED)

        if trace:
            trace.add_span(span)

        return span

    def log_tool_call(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        tool_output: Any,
        **metadata,
    ) -> Span:
        """
        Log a tool call without using a context manager.

        Args:
            tool_name: Name of the tool
            tool_input: Input parameters
            tool_output: Output result
            **metadata: Additional metadata

        Returns:
            Created Span
        """
        trace = get_current_trace()
        parent_span = get_current_span()

        span = Span(
            trace_id=trace.trace_id if trace else "",
            parent_span_id=parent_span.span_id if parent_span else None,
            name=f"tool:{tool_name}",
            span_type=SpanType.TOOL_CALL,
            input_data={"tool": tool_name, "input": tool_input},
            output_data={"output": tool_output},
            metadata=metadata,
        )

        span.finish(SpanStatus.COMPLETED)

        if trace:
            trace.add_span(span)

        return span

    def log_retrieval(
        self,
        query: str,
        results: List[Dict[str, Any]],
        source: str = "unknown",
        **metadata,
    ) -> Span:
        """
        Log a retrieval operation.

        Args:
            query: Search query
            results: Retrieved results
            source: Source identifier
            **metadata: Additional metadata

        Returns:
            Created Span
        """
        trace = get_current_trace()
        parent_span = get_current_span()

        span = Span(
            trace_id=trace.trace_id if trace else "",
            parent_span_id=parent_span.span_id if parent_span else None,
            name=f"retrieval:{source}",
            span_type=SpanType.RETRIEVAL,
            input_data={"query": query, "source": source},
            output_data={"results": results, "count": len(results)},
            metadata=metadata,
        )

        span.finish(SpanStatus.COMPLETED)

        if trace:
            trace.add_span(span)

        return span

    def log_decision(
        self,
        decision: str,
        confidence: float,
        alternatives: Optional[List[Dict[str, Any]]] = None,
        reasoning: Optional[str] = None,
        **metadata,
    ) -> Span:
        """
        Log a decision/reasoning step.

        Args:
            decision: The decision made
            confidence: Confidence score (0-1)
            alternatives: Alternative options considered
            reasoning: Reasoning text
            **metadata: Additional metadata

        Returns:
            Created Span
        """
        trace = get_current_trace()
        parent_span = get_current_span()

        span = Span(
            trace_id=trace.trace_id if trace else "",
            parent_span_id=parent_span.span_id if parent_span else None,
            name="decision",
            span_type=SpanType.REASONING,
            output_data={"decision": decision},
            confidence_score=confidence,
            alternatives_considered=alternatives or [],
            reasoning=reasoning,
            metadata=metadata,
        )

        span.finish(SpanStatus.COMPLETED)

        if trace:
            trace.add_span(span)

        return span
