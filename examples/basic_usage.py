"""
Basic usage example for AgentDiff.

This example demonstrates:
- Setting up tracing
- Capturing agent executions
- Viewing traces
"""

import asyncio

# Import AgentDiff components
import sys
sys.path.insert(0, "../src")

from agentdiff import configure, trace, get_tracer
from agentdiff.storage import SQLiteStore
from agentdiff.core.span import SpanType


async def main():
    # 1. Configure storage
    storage = SQLiteStore("example_traces.db")
    await storage.initialize()

    # 2. Configure the tracer
    tracer = configure(storage)
    await tracer.initialize()

    # 3. Define and trace an agent
    @tracer.trace(name="example-agent", version="1.0.0")
    async def my_agent(query: str):
        """A simple example agent."""

        # Create a span for processing
        with tracer.span("process_query", span_type=SpanType.AGENT_STEP) as span:
            span.input_data = {"query": query}

            # Simulate an LLM call
            with tracer.span("llm_call", span_type=SpanType.LLM_CALL) as llm_span:
                # Log the LLM call details
                tracer.log_llm_call(
                    model="gpt-4",
                    provider="openai",
                    input_messages=[{"role": "user", "content": query}],
                    output_message="This is a simulated response.",
                    token_usage={
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                )

            # Simulate a tool call
            with tracer.span("tool_call", span_type=SpanType.TOOL_CALL) as tool_span:
                tracer.log_tool_call(
                    tool_name="search",
                    tool_input={"query": query},
                    tool_output={"results": ["result1", "result2"]},
                )

            # Simulate a decision
            tracer.log_decision(
                decision="use_search_results",
                confidence=0.85,
                alternatives=[
                    {"option": "ask_clarification", "confidence": 0.10},
                    {"option": "give_up", "confidence": 0.05},
                ],
                reasoning="The search results are relevant to the query.",
            )

            span.output_data = {"result": "Query processed successfully"}

        return "Done!"

    # 4. Run the agent
    print("Running agent...")
    result = await my_agent("What is the weather today?")
    print(f"Result: {result}")

    # 5. List traces
    print("\nRecorded traces:")
    traces = await storage.list_traces(limit=5)
    for t in traces:
        print(f"  - {t.name} (v{t.version}): {len(t.spans)} spans, ${t.total_cost:.4f}")

    # 6. Get the last trace
    if traces:
        last_trace = await storage.get_trace(traces[0].trace_id)
        print(f"\nLast trace details:")
        print(f"  ID: {last_trace.trace_id}")
        print(f"  Name: {last_trace.name}")
        print(f"  Spans: {len(last_trace.spans)}")
        print(f"  Total Tokens: {last_trace.total_tokens}")
        print(f"  Total Cost: ${last_trace.total_cost:.4f}")
        print(f"  Duration: {last_trace.total_duration_ms:.0f}ms")

        print("\n  Span tree:")
        for span in last_trace.spans:
            indent = "    " if span.parent_span_id else "  "
            print(f"{indent}- {span.name} ({span.span_type.value})")

    await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
