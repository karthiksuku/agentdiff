"""
Tests for the AgentDiff tracer.
"""

import pytest
from datetime import datetime

from agentdiff.core.span import Span, SpanType, SpanStatus, TokenUsage
from agentdiff.core.trace import Trace
from agentdiff.core.tracer import AgentDiffTracer, get_current_trace, get_current_span
from agentdiff.storage.sqlite_store import SQLiteStore


@pytest.fixture
async def storage(tmp_path):
    """Create a temporary SQLite storage."""
    db_path = tmp_path / "test.db"
    store = SQLiteStore(str(db_path))
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
async def tracer(storage):
    """Create a tracer with the test storage."""
    return AgentDiffTracer(storage=storage)


class TestSpan:
    """Tests for Span class."""

    def test_span_creation(self):
        """Test basic span creation."""
        span = Span(name="test-span", span_type=SpanType.LLM_CALL)

        assert span.name == "test-span"
        assert span.span_type == SpanType.LLM_CALL
        assert span.status == SpanStatus.RUNNING
        assert span.span_id is not None

    def test_span_finish(self):
        """Test span finish."""
        span = Span(name="test-span")
        span.finish()

        assert span.status == SpanStatus.COMPLETED
        assert span.end_time is not None
        assert span.duration_ms is not None

    def test_span_fail(self):
        """Test span failure."""
        span = Span(name="test-span")
        span.fail("Something went wrong", "ValueError")

        assert span.status == SpanStatus.FAILED
        assert span.error == "Something went wrong"
        assert span.error_type == "ValueError"

    def test_span_to_dict(self):
        """Test span serialization."""
        span = Span(
            name="test-span",
            span_type=SpanType.TOOL_CALL,
            input_data={"query": "test"},
            output_data={"result": "success"},
        )
        span.finish()

        data = span.to_dict()

        assert data["name"] == "test-span"
        assert data["span_type"] == "tool_call"
        assert data["input_data"] == {"query": "test"}
        assert data["output_data"] == {"result": "success"}

    def test_span_from_dict(self):
        """Test span deserialization."""
        data = {
            "span_id": "123",
            "name": "test-span",
            "span_type": "llm_call",
            "status": "completed",
            "input_data": {"query": "test"},
        }

        span = Span.from_dict(data)

        assert span.span_id == "123"
        assert span.name == "test-span"
        assert span.span_type == SpanType.LLM_CALL


class TestTokenUsage:
    """Tests for TokenUsage class."""

    def test_token_usage_creation(self):
        """Test token usage creation."""
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )

        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_calculate_cost(self):
        """Test cost calculation."""
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            total_tokens=1500,
        )

        cost = usage.calculate_cost(
            input_price_per_1k=0.01,
            output_price_per_1k=0.03,
        )

        assert usage.input_cost == 0.01  # 1000/1000 * 0.01
        assert usage.output_cost == 0.015  # 500/1000 * 0.03
        assert cost == pytest.approx(0.025)


class TestTrace:
    """Tests for Trace class."""

    def test_trace_creation(self):
        """Test basic trace creation."""
        trace = Trace(name="test-trace", version="1.0.0")

        assert trace.name == "test-trace"
        assert trace.version == "1.0.0"
        assert trace.trace_id is not None
        assert trace.spans == []

    def test_add_span(self):
        """Test adding spans to trace."""
        trace = Trace(name="test-trace")
        span = Span(name="span-1")

        trace.add_span(span)

        assert len(trace.spans) == 1
        assert span.trace_id == trace.trace_id
        assert trace.root_span_id == span.span_id

    def test_get_span(self):
        """Test getting span by ID."""
        trace = Trace(name="test-trace")
        span = Span(name="span-1")
        trace.add_span(span)

        found = trace.get_span(span.span_id)

        assert found == span

    def test_trace_finish(self):
        """Test trace finish."""
        trace = Trace(name="test-trace")

        span1 = Span(name="span-1")
        span1.token_usage = TokenUsage(total_tokens=100, total_cost=0.01)
        span1.finish()
        trace.add_span(span1)

        span2 = Span(name="span-2")
        span2.token_usage = TokenUsage(total_tokens=200, total_cost=0.02)
        span2.finish()
        trace.add_span(span2)

        trace.finish()

        assert trace.total_tokens == 300
        assert trace.total_cost == pytest.approx(0.03)
        assert trace.status == "completed"

    def test_get_span_tree(self):
        """Test span tree generation."""
        trace = Trace(name="test-trace")

        root = Span(name="root")
        trace.add_span(root)

        child1 = Span(name="child-1", parent_span_id=root.span_id)
        child1.trace_id = trace.trace_id
        trace.spans.append(child1)

        child2 = Span(name="child-2", parent_span_id=root.span_id)
        child2.trace_id = trace.trace_id
        trace.spans.append(child2)

        tree = trace.get_span_tree()

        assert tree["span"] == root
        assert len(tree["children"]) == 2


class TestTracer:
    """Tests for AgentDiffTracer class."""

    @pytest.mark.asyncio
    async def test_trace_context(self, tracer):
        """Test trace context manager."""
        async with tracer.trace_context_async(name="test-agent") as trace:
            assert get_current_trace() == trace
            assert trace.name == "test-agent"

        assert get_current_trace() is None

    @pytest.mark.asyncio
    async def test_span_context(self, tracer):
        """Test span context manager."""
        async with tracer.trace_context_async(name="test-agent") as trace:
            with tracer.span("test-span", span_type=SpanType.LLM_CALL) as span:
                assert get_current_span() == span
                assert span.name == "test-span"
                assert span.span_type == SpanType.LLM_CALL

            assert len(trace.spans) == 1

    @pytest.mark.asyncio
    async def test_nested_spans(self, tracer):
        """Test nested span contexts."""
        async with tracer.trace_context_async(name="test-agent") as trace:
            with tracer.span("parent") as parent:
                with tracer.span("child") as child:
                    assert child.parent_span_id == parent.span_id

            assert len(trace.spans) == 2

    @pytest.mark.asyncio
    async def test_log_llm_call(self, tracer):
        """Test LLM call logging."""
        async with tracer.trace_context_async(name="test-agent"):
            with tracer.span("llm-call") as span:
                tracer.log_llm_call(
                    model="gpt-4",
                    provider="openai",
                    input_messages=[{"role": "user", "content": "Hello"}],
                    output_message="Hi there!",
                    token_usage={"prompt_tokens": 10, "completion_tokens": 5},
                )

                assert span.model == "gpt-4"
                assert span.provider == "openai"
                assert span.span_type == SpanType.LLM_CALL
                assert span.token_usage.input_tokens == 10

    @pytest.mark.asyncio
    async def test_log_tool_call(self, tracer):
        """Test tool call logging."""
        async with tracer.trace_context_async(name="test-agent"):
            with tracer.span("tool-call") as span:
                tracer.log_tool_call(
                    tool_name="search",
                    tool_input={"query": "test"},
                    tool_output={"results": []},
                )

                assert span.span_type == SpanType.TOOL_CALL
                assert span.input_data["tool"] == "search"


class TestStorage:
    """Tests for storage backend."""

    @pytest.mark.asyncio
    async def test_save_and_get_trace(self, storage):
        """Test saving and retrieving a trace."""
        trace = Trace(name="test-trace", version="1.0.0")
        span = Span(name="span-1", span_type=SpanType.LLM_CALL)
        span.finish()
        trace.add_span(span)
        trace.finish()

        await storage.save_trace(trace)

        loaded = await storage.get_trace(trace.trace_id)

        assert loaded is not None
        assert loaded.name == "test-trace"
        assert len(loaded.spans) == 1
        assert loaded.spans[0].name == "span-1"

    @pytest.mark.asyncio
    async def test_list_traces(self, storage):
        """Test listing traces."""
        for i in range(5):
            trace = Trace(name=f"trace-{i}")
            trace.finish()
            await storage.save_trace(trace)

        traces = await storage.list_traces(limit=10)

        assert len(traces) == 5

    @pytest.mark.asyncio
    async def test_delete_trace(self, storage):
        """Test deleting a trace."""
        trace = Trace(name="to-delete")
        trace.finish()
        await storage.save_trace(trace)

        deleted = await storage.delete_trace(trace.trace_id)

        assert deleted is True
        assert await storage.get_trace(trace.trace_id) is None

    @pytest.mark.asyncio
    async def test_get_traces_by_name(self, storage):
        """Test getting traces by name."""
        for i in range(3):
            trace = Trace(name="same-name", version=f"1.{i}.0")
            trace.finish()
            await storage.save_trace(trace)

        traces = await storage.get_traces_by_name("same-name")

        assert len(traces) == 3
