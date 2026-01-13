"""
Example: Comparing different prompt versions.

This example demonstrates:
- Running the same agent with different prompts
- Comparing the execution traces
- Analyzing cost differences
"""

import asyncio

import sys
sys.path.insert(0, "../src")

from agentdiff import configure
from agentdiff.storage import SQLiteStore
from agentdiff.core.span import SpanType, TokenUsage
from agentdiff.diff import StructuralDiffEngine, CostDiffEngine
from agentdiff.diff.renderer import DiffRenderer


async def main():
    # Setup
    storage = SQLiteStore("prompt_comparison.db")
    await storage.initialize()
    tracer = configure(storage)
    await tracer.initialize()

    # Version A: Simple prompt
    @tracer.trace(name="qa-agent", version="1.0.0", branch="main")
    async def agent_v1(question: str):
        with tracer.span("answer", span_type=SpanType.LLM_CALL) as span:
            span.model = "gpt-4"
            span.provider = "openai"
            span.input_data = {
                "messages": [
                    {"role": "user", "content": question}
                ]
            }
            # Simulate response
            span.output_data = {"response": "Simple answer to: " + question}
            span.token_usage = TokenUsage(
                input_tokens=50,
                output_tokens=30,
                total_tokens=80,
                total_cost=0.005,
            )
        return span.output_data["response"]

    # Version B: Chain-of-thought prompt
    @tracer.trace(name="qa-agent", version="2.0.0", branch="main")
    async def agent_v2(question: str):
        # Thinking step
        with tracer.span("think", span_type=SpanType.REASONING) as think_span:
            think_span.model = "gpt-4"
            think_span.input_data = {
                "messages": [
                    {"role": "system", "content": "Think step by step."},
                    {"role": "user", "content": question}
                ]
            }
            think_span.output_data = {"thinking": "Let me break this down..."}
            think_span.token_usage = TokenUsage(
                input_tokens=80,
                output_tokens=100,
                total_tokens=180,
                total_cost=0.012,
            )
            think_span.reasoning = "Breaking down the question into parts"

        # Answer step
        with tracer.span("answer", span_type=SpanType.LLM_CALL) as answer_span:
            answer_span.model = "gpt-4"
            answer_span.input_data = {
                "messages": [
                    {"role": "system", "content": "Based on your thinking, provide an answer."},
                    {"role": "user", "content": question}
                ]
            }
            answer_span.output_data = {"response": "Detailed answer: " + question}
            answer_span.token_usage = TokenUsage(
                input_tokens=100,
                output_tokens=80,
                total_tokens=180,
                total_cost=0.011,
            )

        return answer_span.output_data["response"]

    # Run both versions
    question = "What is the capital of France?"

    print("Running agent v1.0.0 (simple prompt)...")
    result_v1 = await agent_v1(question)
    print(f"  Result: {result_v1}")

    print("\nRunning agent v2.0.0 (chain-of-thought)...")
    result_v2 = await agent_v2(question)
    print(f"  Result: {result_v2}")

    # Get the traces
    traces = await storage.get_traces_by_name("qa-agent", limit=2)

    if len(traces) >= 2:
        trace_v2 = traces[0]  # Newest
        trace_v1 = traces[1]  # Older

        # Compare structurally
        print("\n" + "=" * 60)
        print("STRUCTURAL DIFF")
        print("=" * 60)

        struct_engine = StructuralDiffEngine()
        diff = struct_engine.diff(trace_v1, trace_v2)

        renderer = DiffRenderer(color=True)
        print(renderer.render_terminal(diff))

        # Compare costs
        print("\n" + "=" * 60)
        print("COST COMPARISON")
        print("=" * 60)

        cost_engine = CostDiffEngine()
        cost_diff = cost_engine.compare(trace_v1, trace_v2)

        print(renderer.render_cost_comparison(cost_diff))

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"v1.0.0 (simple):    ${trace_v1.total_cost:.4f}, {trace_v1.total_tokens} tokens")
        print(f"v2.0.0 (CoT):       ${trace_v2.total_cost:.4f}, {trace_v2.total_tokens} tokens")
        print(f"Cost increase:      ${cost_diff.cost_delta:.4f} ({cost_diff.cost_change_percentage:.1f}%)")
        print(f"Token increase:     {diff.token_delta} tokens")

    await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
