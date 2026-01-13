"""
Replay engine for re-executing agent traces.

Allows replaying agent executions from any checkpoint,
with the ability to modify inputs and observe different outcomes.
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio

from ..core.trace import Trace
from ..core.span import Span, SpanType, SpanStatus
from ..storage.base import BaseStorage


class ReplayMode(Enum):
    """Modes for replay execution."""
    EXACT = "exact"  # Replay with exact same responses (mocked)
    LIVE = "live"  # Replay with live LLM calls
    HYBRID = "hybrid"  # Use cached for some, live for modified


@dataclass
class ReplayResult:
    """Result of a replay execution."""
    original_trace: Trace
    replay_trace: Trace

    # Comparison
    diverged: bool = False
    divergence_point: Optional[str] = None  # span_id where divergence occurred

    # Metrics
    original_cost: float = 0.0
    replay_cost: float = 0.0
    cost_delta: float = 0.0

    # Detailed comparison
    span_comparisons: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_trace_id": self.original_trace.trace_id,
            "replay_trace_id": self.replay_trace.trace_id,
            "diverged": self.diverged,
            "divergence_point": self.divergence_point,
            "original_cost": self.original_cost,
            "replay_cost": self.replay_cost,
            "cost_delta": self.cost_delta,
            "span_comparisons": self.span_comparisons,
        }


class ReplayEngine:
    """
    Engine for replaying agent executions.

    Supports:
    - Exact replay with mocked responses
    - Live replay with actual LLM calls
    - Replay from specific checkpoints
    - Modified input replay
    """

    def __init__(
        self,
        storage: BaseStorage,
        mode: ReplayMode = ReplayMode.EXACT,
        llm_callback: Optional[Callable] = None,
        tool_callback: Optional[Callable] = None,
    ):
        """
        Initialize the replay engine.

        Args:
            storage: Storage backend for traces
            mode: Replay mode
            llm_callback: Callback for live LLM calls
            tool_callback: Callback for live tool calls
        """
        self.storage = storage
        self.mode = mode
        self.llm_callback = llm_callback
        self.tool_callback = tool_callback

    async def replay(
        self,
        trace_id: str,
        from_checkpoint: Optional[str] = None,
        modified_inputs: Optional[Dict[str, Any]] = None,
    ) -> ReplayResult:
        """
        Replay a trace execution.

        Args:
            trace_id: ID of the trace to replay
            from_checkpoint: Optional checkpoint ID to start from
            modified_inputs: Optional modified inputs for replay

        Returns:
            ReplayResult with comparison
        """
        # Load original trace
        original_trace = await self.storage.get_trace(trace_id)
        if not original_trace:
            raise ValueError(f"Trace not found: {trace_id}")

        # Create replay trace
        replay_trace = Trace(
            name=f"{original_trace.name}_replay",
            version=original_trace.version,
            branch=original_trace.branch,
            parent_trace_id=original_trace.trace_id,
            metadata={
                **original_trace.metadata,
                "replay_mode": self.mode.value,
                "replay_from_checkpoint": from_checkpoint,
            },
        )

        # Get starting point
        start_span_idx = 0
        if from_checkpoint:
            checkpoint = await self.storage.get_checkpoint(from_checkpoint)
            if checkpoint:
                # Find the span index
                for i, span in enumerate(original_trace.spans):
                    if span.span_id == checkpoint["span_id"]:
                        start_span_idx = i
                        break

        # Replay spans
        result = ReplayResult(
            original_trace=original_trace,
            replay_trace=replay_trace,
        )

        for i, original_span in enumerate(original_trace.spans[start_span_idx:]):
            # Determine if inputs are modified
            span_inputs = original_span.input_data
            if modified_inputs and original_span.span_id in modified_inputs:
                span_inputs = modified_inputs[original_span.span_id]

            # Execute span
            replay_span = await self._execute_span(
                original_span=original_span,
                inputs=span_inputs,
            )

            replay_trace.add_span(replay_span)

            # Compare outputs
            comparison = self._compare_spans(original_span, replay_span)
            result.span_comparisons.append(comparison)

            # Check for divergence
            if comparison["diverged"] and not result.diverged:
                result.diverged = True
                result.divergence_point = replay_span.span_id

        # Finish replay trace
        replay_trace.finish()

        # Calculate costs
        result.original_cost = original_trace.total_cost
        result.replay_cost = replay_trace.total_cost
        result.cost_delta = result.replay_cost - result.original_cost

        # Save replay trace
        await self.storage.save_trace(replay_trace)

        return result

    async def _execute_span(
        self,
        original_span: Span,
        inputs: Dict[str, Any],
    ) -> Span:
        """
        Execute a span during replay.

        Args:
            original_span: The original span
            inputs: Inputs for the span

        Returns:
            Replayed span
        """
        span = Span(
            name=original_span.name,
            span_type=original_span.span_type,
            model=original_span.model,
            provider=original_span.provider,
            input_data=inputs,
            metadata={
                "replay_of": original_span.span_id,
            },
        )

        try:
            if self.mode == ReplayMode.EXACT:
                # Use original output
                span.output_data = original_span.output_data
                span.token_usage = original_span.token_usage

            elif self.mode == ReplayMode.LIVE:
                # Make live call
                if original_span.span_type == SpanType.LLM_CALL:
                    output = await self._live_llm_call(inputs, original_span)
                    span.output_data = output
                elif original_span.span_type == SpanType.TOOL_CALL:
                    output = await self._live_tool_call(inputs, original_span)
                    span.output_data = output
                else:
                    span.output_data = original_span.output_data

            elif self.mode == ReplayMode.HYBRID:
                # Use cached if inputs match, otherwise live
                if inputs == original_span.input_data:
                    span.output_data = original_span.output_data
                    span.token_usage = original_span.token_usage
                else:
                    if original_span.span_type == SpanType.LLM_CALL:
                        output = await self._live_llm_call(inputs, original_span)
                        span.output_data = output
                    else:
                        span.output_data = original_span.output_data

            span.finish(SpanStatus.COMPLETED)

        except Exception as e:
            span.fail(str(e), type(e).__name__)

        return span

    async def _live_llm_call(
        self,
        inputs: Dict[str, Any],
        original_span: Span,
    ) -> Dict[str, Any]:
        """Make a live LLM call."""
        if self.llm_callback:
            return await self.llm_callback(
                model=original_span.model,
                provider=original_span.provider,
                **inputs,
            )

        # Default: return original output with warning
        return {
            **original_span.output_data,
            "_warning": "No LLM callback provided, using original output",
        }

    async def _live_tool_call(
        self,
        inputs: Dict[str, Any],
        original_span: Span,
    ) -> Dict[str, Any]:
        """Make a live tool call."""
        if self.tool_callback:
            tool_name = inputs.get("tool", original_span.name)
            tool_input = inputs.get("input", {})
            return await self.tool_callback(
                tool_name=tool_name,
                tool_input=tool_input,
            )

        # Default: return original output with warning
        return {
            **original_span.output_data,
            "_warning": "No tool callback provided, using original output",
        }

    def _compare_spans(
        self,
        original: Span,
        replay: Span,
    ) -> Dict[str, Any]:
        """
        Compare original and replay spans.

        Args:
            original: Original span
            replay: Replay span

        Returns:
            Comparison dictionary
        """
        output_match = original.output_data == replay.output_data

        return {
            "original_span_id": original.span_id,
            "replay_span_id": replay.span_id,
            "span_name": original.name,
            "diverged": not output_match,
            "input_modified": original.input_data != replay.input_data,
            "output_match": output_match,
        }

    async def replay_from_span(
        self,
        trace_id: str,
        span_id: str,
        modified_input: Optional[Dict[str, Any]] = None,
    ) -> ReplayResult:
        """
        Replay from a specific span.

        Args:
            trace_id: ID of the trace
            span_id: ID of the span to start from
            modified_input: Optional modified input for the starting span

        Returns:
            ReplayResult
        """
        modified_inputs = None
        if modified_input:
            modified_inputs = {span_id: modified_input}

        # Find or create checkpoint
        checkpoints = await self.storage.list_checkpoints(trace_id)
        checkpoint_id = None

        for cp in checkpoints:
            if cp["span_id"] == span_id:
                checkpoint_id = cp["checkpoint_id"]
                break

        if not checkpoint_id:
            # Create checkpoint
            checkpoint_id = await self.storage.create_checkpoint(
                trace_id=trace_id,
                span_id=span_id,
                name=f"replay_checkpoint_{span_id[:8]}",
                state_snapshot={},
            )

        return await self.replay(
            trace_id=trace_id,
            from_checkpoint=checkpoint_id,
            modified_inputs=modified_inputs,
        )

    async def what_if(
        self,
        trace_id: str,
        span_id: str,
        alternative_output: Dict[str, Any],
    ) -> ReplayResult:
        """
        Execute a "what-if" scenario.

        Replay the trace as if a specific span had a different output.

        Args:
            trace_id: ID of the trace
            span_id: ID of the span to modify
            alternative_output: Alternative output for the span

        Returns:
            ReplayResult showing the impact
        """
        # Load original trace
        original_trace = await self.storage.get_trace(trace_id)
        if not original_trace:
            raise ValueError(f"Trace not found: {trace_id}")

        # Find the span index
        span_idx = None
        for i, span in enumerate(original_trace.spans):
            if span.span_id == span_id:
                span_idx = i
                break

        if span_idx is None:
            raise ValueError(f"Span not found: {span_id}")

        # Create modified trace
        replay_trace = Trace(
            name=f"{original_trace.name}_whatif",
            version=original_trace.version,
            parent_trace_id=original_trace.trace_id,
            metadata={
                "what_if": True,
                "modified_span": span_id,
            },
        )

        result = ReplayResult(
            original_trace=original_trace,
            replay_trace=replay_trace,
        )

        # Copy spans up to the modified span
        for span in original_trace.spans[:span_idx]:
            replay_trace.add_span(span)

        # Add modified span
        modified_span = Span(
            name=original_trace.spans[span_idx].name,
            span_type=original_trace.spans[span_idx].span_type,
            input_data=original_trace.spans[span_idx].input_data,
            output_data=alternative_output,
            metadata={"what_if_modified": True},
        )
        modified_span.finish()
        replay_trace.add_span(modified_span)

        # Replay remaining spans in HYBRID mode (live calls since context changed)
        original_mode = self.mode
        self.mode = ReplayMode.HYBRID

        for original_span in original_trace.spans[span_idx + 1:]:
            replay_span = await self._execute_span(
                original_span=original_span,
                inputs=original_span.input_data,
            )
            replay_trace.add_span(replay_span)

            comparison = self._compare_spans(original_span, replay_span)
            result.span_comparisons.append(comparison)

            if comparison["diverged"] and not result.diverged:
                result.diverged = True
                result.divergence_point = replay_span.span_id

        self.mode = original_mode

        replay_trace.finish()
        await self.storage.save_trace(replay_trace)

        result.original_cost = original_trace.total_cost
        result.replay_cost = replay_trace.total_cost
        result.cost_delta = result.replay_cost - result.original_cost

        return result
