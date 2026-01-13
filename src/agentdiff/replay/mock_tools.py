"""
Mock tool providers for replay functionality.

Provides mocked responses for tools during replay to ensure
deterministic execution without side effects.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import json
import re


@dataclass
class MockResponse:
    """A mocked response for a tool or LLM call."""
    pattern: str  # Regex pattern to match input
    response: Any
    delay_ms: float = 0.0
    times: int = -1  # -1 for unlimited

    # Tracking
    call_count: int = 0

    def matches(self, input_str: str) -> bool:
        """Check if input matches this mock."""
        return bool(re.search(self.pattern, input_str))

    def can_respond(self) -> bool:
        """Check if this mock can still respond."""
        return self.times == -1 or self.call_count < self.times

    def get_response(self) -> Any:
        """Get the response and increment counter."""
        self.call_count += 1
        return self.response


class MockToolProvider:
    """
    Provides mocked tool responses for replay.

    Allows registering mock responses for specific tool calls,
    enabling deterministic replay without external dependencies.
    """

    def __init__(self):
        """Initialize the mock provider."""
        self.tool_mocks: Dict[str, List[MockResponse]] = {}
        self.llm_mocks: List[MockResponse] = []
        self.default_tool_response: Optional[Callable] = None
        self.default_llm_response: Optional[Callable] = None
        self.call_history: List[Dict[str, Any]] = []

    def register_tool_mock(
        self,
        tool_name: str,
        pattern: str,
        response: Any,
        times: int = -1,
    ) -> None:
        """
        Register a mock response for a tool.

        Args:
            tool_name: Name of the tool
            pattern: Regex pattern to match input
            response: Response to return
            times: Number of times to respond (-1 for unlimited)
        """
        if tool_name not in self.tool_mocks:
            self.tool_mocks[tool_name] = []

        self.tool_mocks[tool_name].append(MockResponse(
            pattern=pattern,
            response=response,
            times=times,
        ))

    def register_llm_mock(
        self,
        pattern: str,
        response: str,
        times: int = -1,
    ) -> None:
        """
        Register a mock response for LLM calls.

        Args:
            pattern: Regex pattern to match input
            response: Response to return
            times: Number of times to respond
        """
        self.llm_mocks.append(MockResponse(
            pattern=pattern,
            response=response,
            times=times,
        ))

    def set_default_tool_response(
        self,
        handler: Callable[[str, Dict[str, Any]], Any],
    ) -> None:
        """
        Set default handler for unmatched tool calls.

        Args:
            handler: Function(tool_name, input) -> response
        """
        self.default_tool_response = handler

    def set_default_llm_response(
        self,
        handler: Callable[[List[Dict]], str],
    ) -> None:
        """
        Set default handler for unmatched LLM calls.

        Args:
            handler: Function(messages) -> response
        """
        self.default_llm_response = handler

    async def call_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> Any:
        """
        Call a mocked tool.

        Args:
            tool_name: Name of the tool
            tool_input: Input for the tool

        Returns:
            Mocked response
        """
        input_str = json.dumps(tool_input)

        # Record call
        self.call_history.append({
            "type": "tool",
            "tool_name": tool_name,
            "input": tool_input,
        })

        # Check for matching mock
        if tool_name in self.tool_mocks:
            for mock in self.tool_mocks[tool_name]:
                if mock.matches(input_str) and mock.can_respond():
                    return mock.get_response()

        # Use default handler
        if self.default_tool_response:
            return self.default_tool_response(tool_name, tool_input)

        # Return empty response
        return {"_mock": True, "_warning": f"No mock for tool: {tool_name}"}

    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> str:
        """
        Call a mocked LLM.

        Args:
            messages: Input messages
            model: Optional model identifier

        Returns:
            Mocked response
        """
        input_str = json.dumps(messages)

        # Record call
        self.call_history.append({
            "type": "llm",
            "messages": messages,
            "model": model,
        })

        # Check for matching mock
        for mock in self.llm_mocks:
            if mock.matches(input_str) and mock.can_respond():
                return mock.get_response()

        # Use default handler
        if self.default_llm_response:
            return self.default_llm_response(messages)

        # Return empty response
        return "[MOCK RESPONSE - No mock registered]"

    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get the call history."""
        return self.call_history

    def clear_history(self) -> None:
        """Clear the call history."""
        self.call_history = []

    def reset_mocks(self) -> None:
        """Reset all mock call counts."""
        for mocks in self.tool_mocks.values():
            for mock in mocks:
                mock.call_count = 0
        for mock in self.llm_mocks:
            mock.call_count = 0

    @classmethod
    def from_trace(cls, trace: "Trace") -> "MockToolProvider":
        """
        Create a mock provider from an existing trace.

        Extracts tool and LLM calls from the trace and creates
        mocks that will replay the exact same responses.

        Args:
            trace: The trace to extract mocks from

        Returns:
            MockToolProvider configured with trace responses
        """
        from ..core.span import SpanType

        provider = cls()

        for span in trace.spans:
            if span.span_type == SpanType.TOOL_CALL:
                tool_name = span.input_data.get("tool", span.name)
                # Use exact input as pattern (escaped)
                input_str = json.dumps(span.input_data.get("input", {}))
                pattern = re.escape(input_str)

                provider.register_tool_mock(
                    tool_name=tool_name,
                    pattern=pattern,
                    response=span.output_data.get("output", span.output_data),
                    times=1,  # Only respond once per call
                )

            elif span.span_type == SpanType.LLM_CALL:
                messages = span.input_data.get("messages", [])
                if messages:
                    # Use last message content as pattern
                    last_msg = messages[-1].get("content", "")
                    pattern = re.escape(last_msg[:100])  # First 100 chars

                    provider.register_llm_mock(
                        pattern=pattern,
                        response=span.output_data.get("response", ""),
                        times=1,
                    )

        return provider


class DeterministicMockProvider(MockToolProvider):
    """
    A mock provider that ensures deterministic ordering.

    Responses are returned in the exact order they were registered,
    regardless of input matching.
    """

    def __init__(self):
        super().__init__()
        self.tool_queue: Dict[str, List[Any]] = {}
        self.llm_queue: List[str] = []

    def queue_tool_response(self, tool_name: str, response: Any) -> None:
        """Queue a response for a tool."""
        if tool_name not in self.tool_queue:
            self.tool_queue[tool_name] = []
        self.tool_queue[tool_name].append(response)

    def queue_llm_response(self, response: str) -> None:
        """Queue a response for LLM calls."""
        self.llm_queue.append(response)

    async def call_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> Any:
        """Get the next queued response for a tool."""
        self.call_history.append({
            "type": "tool",
            "tool_name": tool_name,
            "input": tool_input,
        })

        if tool_name in self.tool_queue and self.tool_queue[tool_name]:
            return self.tool_queue[tool_name].pop(0)

        return await super().call_tool(tool_name, tool_input)

    async def call_llm(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
    ) -> str:
        """Get the next queued LLM response."""
        self.call_history.append({
            "type": "llm",
            "messages": messages,
            "model": model,
        })

        if self.llm_queue:
            return self.llm_queue.pop(0)

        return await super().call_llm(messages, model)
