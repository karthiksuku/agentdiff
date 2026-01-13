"""
Tests for the replay engine.
"""

import pytest

from agentdiff.core.span import Span, SpanType, SpanStatus
from agentdiff.core.trace import Trace
from agentdiff.replay.engine import ReplayEngine, ReplayMode, ReplayResult
from agentdiff.replay.checkpoint import CheckpointManager, Checkpoint
from agentdiff.replay.mock_tools import MockToolProvider, DeterministicMockProvider
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
def sample_trace():
    """Create a sample trace for testing."""
    trace = Trace(name="test-agent", version="1.0.0")

    # Add some spans
    span1 = Span(name="process", span_type=SpanType.AGENT_STEP)
    span1.input_data = {"query": "test query"}
    span1.output_data = {"result": "processed"}
    span1.finish()
    trace.add_span(span1)

    span2 = Span(name="llm-call", span_type=SpanType.LLM_CALL, parent_span_id=span1.span_id)
    span2.trace_id = trace.trace_id
    span2.model = "gpt-4"
    span2.input_data = {"messages": [{"role": "user", "content": "test"}]}
    span2.output_data = {"response": "test response"}
    span2.finish()
    trace.spans.append(span2)

    span3 = Span(name="tool-call", span_type=SpanType.TOOL_CALL, parent_span_id=span1.span_id)
    span3.trace_id = trace.trace_id
    span3.input_data = {"tool": "search", "input": {"q": "test"}}
    span3.output_data = {"output": ["result1", "result2"]}
    span3.finish()
    trace.spans.append(span3)

    trace.finish()
    return trace


class TestReplayEngine:
    """Tests for ReplayEngine."""

    @pytest.mark.asyncio
    async def test_exact_replay(self, storage, sample_trace):
        """Test exact replay mode."""
        await storage.save_trace(sample_trace)

        engine = ReplayEngine(storage, mode=ReplayMode.EXACT)
        result = await engine.replay(sample_trace.trace_id)

        assert result.original_trace.trace_id == sample_trace.trace_id
        assert result.replay_trace.trace_id != sample_trace.trace_id
        assert not result.diverged

    @pytest.mark.asyncio
    async def test_replay_from_span(self, storage, sample_trace):
        """Test replay from specific span."""
        await storage.save_trace(sample_trace)

        span_id = sample_trace.spans[1].span_id  # LLM call span

        engine = ReplayEngine(storage, mode=ReplayMode.EXACT)
        result = await engine.replay_from_span(sample_trace.trace_id, span_id)

        assert result.original_trace.trace_id == sample_trace.trace_id

    @pytest.mark.asyncio
    async def test_replay_with_modified_input(self, storage, sample_trace):
        """Test replay with modified inputs."""
        await storage.save_trace(sample_trace)

        span_id = sample_trace.spans[0].span_id
        modified_input = {"query": "modified query"}

        engine = ReplayEngine(storage, mode=ReplayMode.EXACT)
        result = await engine.replay(
            sample_trace.trace_id,
            modified_inputs={span_id: modified_input},
        )

        # With exact mode and modified input, we should still get original output
        assert result.original_trace.trace_id == sample_trace.trace_id


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, storage, sample_trace):
        """Test checkpoint creation."""
        await storage.save_trace(sample_trace)

        manager = CheckpointManager(storage)
        span = sample_trace.spans[0]

        checkpoint = await manager.create_checkpoint(
            trace=sample_trace,
            span=span,
            name="test-checkpoint",
            state={"custom": "data"},
        )

        assert checkpoint.checkpoint_id is not None
        assert checkpoint.trace_id == sample_trace.trace_id
        assert checkpoint.span_id == span.span_id
        assert checkpoint.name == "test-checkpoint"
        assert "custom" in checkpoint.state_snapshot

    @pytest.mark.asyncio
    async def test_get_checkpoint(self, storage, sample_trace):
        """Test checkpoint retrieval."""
        await storage.save_trace(sample_trace)

        manager = CheckpointManager(storage)
        span = sample_trace.spans[0]

        created = await manager.create_checkpoint(
            trace=sample_trace,
            span=span,
            name="test",
        )

        loaded = await manager.get_checkpoint(created.checkpoint_id)

        assert loaded is not None
        assert loaded.checkpoint_id == created.checkpoint_id
        assert loaded.span_id == span.span_id

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, storage, sample_trace):
        """Test listing checkpoints."""
        await storage.save_trace(sample_trace)

        manager = CheckpointManager(storage)

        # Create multiple checkpoints
        for span in sample_trace.spans[:2]:
            await manager.create_checkpoint(
                trace=sample_trace,
                span=span,
            )

        checkpoints = await manager.list_checkpoints(sample_trace.trace_id)

        assert len(checkpoints) == 2

    @pytest.mark.asyncio
    async def test_auto_checkpoint(self, storage, sample_trace):
        """Test automatic checkpoint creation."""
        await storage.save_trace(sample_trace)

        manager = CheckpointManager(storage)
        checkpoints = await manager.auto_checkpoint(sample_trace)

        # Should create checkpoints for tool_call and llm_call spans
        assert len(checkpoints) >= 2


class TestMockToolProvider:
    """Tests for MockToolProvider."""

    @pytest.mark.asyncio
    async def test_tool_mock(self):
        """Test tool mocking."""
        provider = MockToolProvider()
        provider.register_tool_mock(
            tool_name="search",
            pattern=".*test.*",
            response={"results": ["mock result"]},
        )

        result = await provider.call_tool("search", {"query": "test query"})

        assert result == {"results": ["mock result"]}

    @pytest.mark.asyncio
    async def test_llm_mock(self):
        """Test LLM mocking."""
        provider = MockToolProvider()
        provider.register_llm_mock(
            pattern=".*hello.*",
            response="Hi there!",
        )

        result = await provider.call_llm([{"role": "user", "content": "hello"}])

        assert result == "Hi there!"

    @pytest.mark.asyncio
    async def test_mock_times_limit(self):
        """Test mock response limit."""
        provider = MockToolProvider()
        provider.register_tool_mock(
            tool_name="limited",
            pattern=".*",
            response="limited response",
            times=2,
        )

        # First two calls should work
        await provider.call_tool("limited", {})
        await provider.call_tool("limited", {})

        # Third call should fall through to default
        result = await provider.call_tool("limited", {})
        assert "_mock" in result

    def test_call_history(self):
        """Test call history tracking."""
        import asyncio

        provider = MockToolProvider()
        provider.register_tool_mock("test", ".*", "response")

        asyncio.get_event_loop().run_until_complete(
            provider.call_tool("test", {"key": "value"})
        )

        history = provider.get_call_history()

        assert len(history) == 1
        assert history[0]["type"] == "tool"
        assert history[0]["tool_name"] == "test"


class TestDeterministicMockProvider:
    """Tests for DeterministicMockProvider."""

    @pytest.mark.asyncio
    async def test_queued_responses(self):
        """Test deterministic queued responses."""
        provider = DeterministicMockProvider()
        provider.queue_tool_response("test", "first")
        provider.queue_tool_response("test", "second")
        provider.queue_tool_response("test", "third")

        results = []
        for _ in range(3):
            result = await provider.call_tool("test", {})
            results.append(result)

        assert results == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_queued_llm_responses(self):
        """Test deterministic LLM responses."""
        provider = DeterministicMockProvider()
        provider.queue_llm_response("response 1")
        provider.queue_llm_response("response 2")

        r1 = await provider.call_llm([])
        r2 = await provider.call_llm([])

        assert r1 == "response 1"
        assert r2 == "response 2"
