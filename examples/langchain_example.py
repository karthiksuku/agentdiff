"""
Example: Using AgentDiff with LangChain.

This example demonstrates:
- Integrating AgentDiff with LangChain
- Automatic capture of LangChain operations
- Viewing captured traces
"""

import asyncio

import sys
sys.path.insert(0, "../src")

from agentdiff import configure
from agentdiff.storage import SQLiteStore
from agentdiff.capture import LangChainCallback


async def main():
    """
    LangChain Integration Example.

    This example shows how to use AgentDiff with LangChain.
    The callback captures all LangChain operations automatically.
    """

    print("LangChain Integration Example")
    print("=" * 50)

    # Setup storage
    storage = SQLiteStore("langchain_traces.db")
    await storage.initialize()
    tracer = configure(storage)
    await tracer.initialize()

    # Create LangChain callback
    callback = LangChainCallback()

    print("\nTo use with LangChain:")
    print("-" * 40)
    print("""
# Import the callback
from agentdiff.capture import LangChainCallback

# Create the callback
callback = LangChainCallback()

# Use with ChatOpenAI
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4",
    callbacks=[callback]
)

# Or use with chains
from langchain.chains import LLMChain

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.invoke(
    {"input": "your query"},
    config={"callbacks": [callback]}
)

# Or use with agents
from langchain.agents import create_react_agent

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[callback]
)
    """)

    print("\nCaptures the following operations:")
    print("-" * 40)
    print("  - LLM calls (on_llm_start/end)")
    print("  - Chat model calls (on_chat_model_start)")
    print("  - Chain executions (on_chain_start/end)")
    print("  - Tool calls (on_tool_start/end)")
    print("  - Retriever operations (on_retriever_start/end)")
    print("  - Agent actions and finishes")

    # Simulate what the callback captures
    print("\nSimulating LangChain operations...")
    print("-" * 40)

    # Simulate a chain execution trace
    async with tracer.trace_context_async(name="langchain-agent") as trace:
        # Simulate chain start
        with tracer.span("chain:QAChain") as chain_span:
            chain_span.input_data = {"question": "What is AI?"}

            # Simulate retriever
            with tracer.span("retrieval:VectorStore") as retriever_span:
                retriever_span.input_data = {"query": "What is AI?"}
                retriever_span.output_data = {
                    "documents": [
                        {"content": "AI is artificial intelligence..."},
                        {"content": "Machine learning is a subset of AI..."},
                    ],
                    "count": 2,
                }

            # Simulate LLM call
            with tracer.span("chat:gpt-4") as llm_span:
                llm_span.model = "gpt-4"
                llm_span.provider = "openai"
                tracer.log_llm_call(
                    model="gpt-4",
                    provider="openai",
                    input_messages=[
                        {"role": "system", "content": "Answer based on context."},
                        {"role": "user", "content": "What is AI?"},
                    ],
                    output_message="AI is the simulation of human intelligence...",
                    token_usage={"prompt_tokens": 150, "completion_tokens": 100},
                )

            chain_span.output_data = {
                "output": "AI is the simulation of human intelligence..."
            }

    print("Simulated trace created!")

    # Show the trace
    traces = await storage.list_traces(limit=1)
    if traces:
        trace = await storage.get_trace(traces[0].trace_id)
        print(f"\nTrace: {trace.name}")
        print(f"  Spans: {len(trace.spans)}")
        for span in trace.spans:
            indent = "    " if span.parent_span_id else "  "
            print(f"{indent}- {span.name} ({span.span_type.value})")

    await storage.close()


if __name__ == "__main__":
    asyncio.run(main())
