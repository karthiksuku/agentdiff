"""
CrewAI callback handler for automatic tracing.
"""

from typing import Any, Dict, List, Optional
import logging

from ..core.tracer import get_current_trace, get_current_span
from ..core.span import Span, SpanType, SpanStatus

logger = logging.getLogger(__name__)


class CrewAICallback:
    """
    CrewAI callback handler for AgentDiff tracing.

    Usage:
        from agentdiff.capture import CrewAICallback
        from crewai import Crew, Agent, Task

        callback = CrewAICallback()

        crew = Crew(
            agents=[...],
            tasks=[...],
            callbacks=[callback],
        )
    """

    def __init__(self):
        """Initialize the callback handler."""
        self._task_spans: Dict[str, Span] = {}
        self._agent_spans: Dict[str, Span] = {}

    def _get_or_create_span(
        self,
        key: str,
        name: str,
        span_type: SpanType,
        spans_dict: Dict[str, Span],
    ) -> Span:
        """Get or create a span."""
        if key in spans_dict:
            return spans_dict[key]

        trace = get_current_trace()
        parent_span = get_current_span()

        span = Span(
            trace_id=trace.trace_id if trace else "",
            parent_span_id=parent_span.span_id if parent_span else None,
            name=name,
            span_type=span_type,
        )

        if trace:
            trace.add_span(span)

        spans_dict[key] = span
        return span

    # Crew callbacks
    def on_crew_start(
        self,
        crew: Any,
        inputs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Called when crew starts."""
        trace = get_current_trace()
        if trace:
            trace.metadata["crew_id"] = getattr(crew, "id", "unknown")
            trace.metadata["crew_agents"] = len(getattr(crew, "agents", []))
            trace.metadata["crew_tasks"] = len(getattr(crew, "tasks", []))
            if inputs:
                trace.metadata["inputs"] = inputs

    def on_crew_end(
        self,
        crew: Any,
        output: Any,
    ) -> None:
        """Called when crew finishes."""
        trace = get_current_trace()
        if trace:
            trace.metadata["crew_output"] = str(output)[:1000]  # Truncate

    def on_crew_error(
        self,
        crew: Any,
        error: BaseException,
    ) -> None:
        """Called when crew errors."""
        trace = get_current_trace()
        if trace:
            trace.error = str(error)
            trace.status = "failed"

    # Task callbacks
    def on_task_start(
        self,
        task: Any,
        agent: Any,
    ) -> None:
        """Called when task starts."""
        task_id = getattr(task, "id", str(id(task)))
        task_desc = getattr(task, "description", "unknown")[:50]
        agent_role = getattr(agent, "role", "unknown")

        span = self._get_or_create_span(
            task_id,
            f"task:{task_desc}",
            SpanType.AGENT_STEP,
            self._task_spans,
        )

        span.input_data = {
            "task_description": getattr(task, "description", ""),
            "expected_output": getattr(task, "expected_output", ""),
            "agent_role": agent_role,
            "agent_goal": getattr(agent, "goal", ""),
        }

    def on_task_end(
        self,
        task: Any,
        output: Any,
    ) -> None:
        """Called when task finishes."""
        task_id = getattr(task, "id", str(id(task)))
        span = self._task_spans.get(task_id)

        if span:
            span.output_data = {
                "output": str(output)[:2000],  # Truncate
                "raw": getattr(output, "raw", None),
            }
            span.finish(SpanStatus.COMPLETED)
            del self._task_spans[task_id]

    def on_task_error(
        self,
        task: Any,
        error: BaseException,
    ) -> None:
        """Called when task errors."""
        task_id = getattr(task, "id", str(id(task)))
        span = self._task_spans.get(task_id)

        if span:
            span.error = str(error)
            span.error_type = type(error).__name__
            span.finish(SpanStatus.FAILED)
            del self._task_spans[task_id]

    # Agent callbacks
    def on_agent_start(
        self,
        agent: Any,
        task: Any,
    ) -> None:
        """Called when agent starts working on a task."""
        agent_role = getattr(agent, "role", "unknown")
        agent_key = f"{agent_role}_{id(task)}"

        span = self._get_or_create_span(
            agent_key,
            f"agent:{agent_role}",
            SpanType.REASONING,
            self._agent_spans,
        )

        span.input_data = {
            "role": agent_role,
            "goal": getattr(agent, "goal", ""),
            "backstory": getattr(agent, "backstory", "")[:500],
            "task": getattr(task, "description", "")[:500],
        }

    def on_agent_end(
        self,
        agent: Any,
        task: Any,
        output: Any,
    ) -> None:
        """Called when agent finishes."""
        agent_role = getattr(agent, "role", "unknown")
        agent_key = f"{agent_role}_{id(task)}"
        span = self._agent_spans.get(agent_key)

        if span:
            span.output_data = {
                "output": str(output)[:2000],
            }
            span.finish(SpanStatus.COMPLETED)
            del self._agent_spans[agent_key]

    def on_agent_action(
        self,
        agent: Any,
        action: str,
        action_input: Any,
    ) -> None:
        """Called when agent takes an action."""
        agent_role = getattr(agent, "role", "unknown")

        # Find the agent span
        for key, span in self._agent_spans.items():
            if key.startswith(agent_role):
                if "actions" not in span.metadata:
                    span.metadata["actions"] = []
                span.metadata["actions"].append({
                    "action": action,
                    "input": str(action_input)[:500],
                })
                break

    def on_tool_use(
        self,
        agent: Any,
        tool_name: str,
        tool_input: Any,
        tool_output: Any,
    ) -> None:
        """Called when agent uses a tool."""
        trace = get_current_trace()
        if not trace:
            return

        parent_span = get_current_span()

        span = Span(
            trace_id=trace.trace_id,
            parent_span_id=parent_span.span_id if parent_span else None,
            name=f"tool:{tool_name}",
            span_type=SpanType.TOOL_CALL,
            input_data={
                "tool": tool_name,
                "input": str(tool_input)[:1000],
                "agent": getattr(agent, "role", "unknown"),
            },
            output_data={
                "output": str(tool_output)[:2000],
            },
        )

        span.finish(SpanStatus.COMPLETED)
        trace.add_span(span)

    def on_delegation(
        self,
        from_agent: Any,
        to_agent: Any,
        task: str,
    ) -> None:
        """Called when work is delegated between agents."""
        trace = get_current_trace()
        if not trace:
            return

        parent_span = get_current_span()

        span = Span(
            trace_id=trace.trace_id,
            parent_span_id=parent_span.span_id if parent_span else None,
            name="delegation",
            span_type=SpanType.PLANNING,
            input_data={
                "from_agent": getattr(from_agent, "role", "unknown"),
                "to_agent": getattr(to_agent, "role", "unknown"),
                "task": task[:500],
            },
        )

        span.finish(SpanStatus.COMPLETED)
        trace.add_span(span)

    def on_thought(
        self,
        agent: Any,
        thought: str,
    ) -> None:
        """Called when agent has a thought/reasoning step."""
        agent_role = getattr(agent, "role", "unknown")

        # Find the agent span and add thought
        for key, span in self._agent_spans.items():
            if key.startswith(agent_role):
                if "thoughts" not in span.metadata:
                    span.metadata["thoughts"] = []
                span.metadata["thoughts"].append(thought[:500])
                span.reasoning = thought
                break
