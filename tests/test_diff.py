"""
Tests for the diff engine.
"""

import pytest

from agentdiff.core.span import Span, SpanType, SpanStatus, TokenUsage
from agentdiff.core.trace import Trace
from agentdiff.diff.structural_diff import StructuralDiffEngine, DiffType
from agentdiff.diff.cost_diff import CostDiffEngine


class TestStructuralDiff:
    """Tests for StructuralDiffEngine."""

    def create_trace(self, name: str, spans_config: list) -> Trace:
        """Helper to create test traces."""
        trace = Trace(name=name)
        for config in spans_config:
            span = Span(
                name=config["name"],
                span_type=config.get("type", SpanType.CUSTOM),
            )
            if "tokens" in config:
                span.token_usage = TokenUsage(
                    total_tokens=config["tokens"],
                    total_cost=config.get("cost", 0),
                )
            span.finish()
            trace.add_span(span)
        trace.finish()
        return trace

    def test_identical_traces(self):
        """Test diff of identical traces."""
        spans = [
            {"name": "span-1", "type": SpanType.LLM_CALL},
            {"name": "span-2", "type": SpanType.TOOL_CALL},
        ]

        trace_a = self.create_trace("test", spans)
        trace_b = self.create_trace("test", spans)

        engine = StructuralDiffEngine()
        diff = engine.diff(trace_a, trace_b)

        assert diff.structural_similarity > 0.9
        assert diff.total_divergences == 0
        assert diff.token_delta == 0

    def test_added_span(self):
        """Test diff with added span."""
        spans_a = [{"name": "span-1"}]
        spans_b = [{"name": "span-1"}, {"name": "span-2"}]

        trace_a = self.create_trace("test", spans_a)
        trace_b = self.create_trace("test", spans_b)

        engine = StructuralDiffEngine()
        diff = engine.diff(trace_a, trace_b)

        added = [d for d in diff.span_diffs if d.diff_type == DiffType.ADDED]
        assert len(added) == 1
        assert added[0].span_b.name == "span-2"

    def test_removed_span(self):
        """Test diff with removed span."""
        spans_a = [{"name": "span-1"}, {"name": "span-2"}]
        spans_b = [{"name": "span-1"}]

        trace_a = self.create_trace("test", spans_a)
        trace_b = self.create_trace("test", spans_b)

        engine = StructuralDiffEngine()
        diff = engine.diff(trace_a, trace_b)

        removed = [d for d in diff.span_diffs if d.diff_type == DiffType.REMOVED]
        assert len(removed) == 1
        assert removed[0].span_a.name == "span-2"

    def test_modified_span(self):
        """Test diff with modified span."""
        trace_a = Trace(name="test")
        span_a = Span(name="span-1", span_type=SpanType.LLM_CALL)
        span_a.input_data = {"query": "original"}
        span_a.finish()
        trace_a.add_span(span_a)
        trace_a.finish()

        trace_b = Trace(name="test")
        span_b = Span(name="span-1", span_type=SpanType.LLM_CALL)
        span_b.input_data = {"query": "modified"}
        span_b.finish()
        trace_b.add_span(span_b)
        trace_b.finish()

        engine = StructuralDiffEngine()
        diff = engine.diff(trace_a, trace_b)

        modified = [d for d in diff.span_diffs if d.diff_type == DiffType.MODIFIED]
        assert len(modified) == 1
        assert "input_data" in modified[0].field_changes

    def test_token_delta(self):
        """Test token delta calculation."""
        spans_a = [{"name": "span-1", "tokens": 100}]
        spans_b = [{"name": "span-1", "tokens": 150}]

        trace_a = self.create_trace("test", spans_a)
        trace_b = self.create_trace("test", spans_b)

        engine = StructuralDiffEngine()
        diff = engine.diff(trace_a, trace_b)

        assert diff.token_delta == 50

    def test_cost_delta(self):
        """Test cost delta calculation."""
        spans_a = [{"name": "span-1", "tokens": 100, "cost": 0.01}]
        spans_b = [{"name": "span-1", "tokens": 100, "cost": 0.02}]

        trace_a = self.create_trace("test", spans_a)
        trace_b = self.create_trace("test", spans_b)

        engine = StructuralDiffEngine()
        diff = engine.diff(trace_a, trace_b)

        assert diff.cost_delta == pytest.approx(0.01)

    def test_divergence_detection(self):
        """Test divergence point detection."""
        trace_a = Trace(name="test")
        span_a = Span(name="decision", span_type=SpanType.TOOL_CALL)
        span_a.input_data = {"tool": "tool_a"}
        span_a.finish()
        trace_a.add_span(span_a)
        trace_a.finish()

        trace_b = Trace(name="test")
        span_b = Span(name="decision", span_type=SpanType.TOOL_CALL)
        span_b.input_data = {"tool": "tool_b"}  # Different tool
        span_b.finish()
        trace_b.add_span(span_b)
        trace_b.finish()

        engine = StructuralDiffEngine()
        diff = engine.diff(trace_a, trace_b)

        assert diff.total_divergences > 0
        assert len(diff.divergence_points) > 0

    def test_to_dict(self):
        """Test diff serialization."""
        spans = [{"name": "span-1"}]
        trace_a = self.create_trace("test-a", spans)
        trace_b = self.create_trace("test-b", spans)

        engine = StructuralDiffEngine()
        diff = engine.diff(trace_a, trace_b)

        data = diff.to_dict()

        assert "trace_a_id" in data
        assert "trace_b_id" in data
        assert "structural_similarity" in data
        assert "span_diffs" in data


class TestCostDiff:
    """Tests for CostDiffEngine."""

    def test_analyze_trace(self):
        """Test trace cost analysis."""
        trace = Trace(name="test")

        span1 = Span(name="llm-1", span_type=SpanType.LLM_CALL, model="gpt-4")
        span1.token_usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            total_cost=0.01,
        )
        span1.finish()
        trace.add_span(span1)

        span2 = Span(name="llm-2", span_type=SpanType.LLM_CALL, model="gpt-4")
        span2.token_usage = TokenUsage(
            input_tokens=200,
            output_tokens=100,
            total_tokens=300,
            total_cost=0.02,
        )
        span2.finish()
        trace.add_span(span2)

        trace.finish()

        engine = CostDiffEngine()
        breakdown = engine.analyze_trace(trace)

        assert breakdown.total_tokens == 450
        assert breakdown.total_cost == pytest.approx(0.03)
        assert "gpt-4" in breakdown.cost_by_model
        assert "llm_call" in breakdown.cost_by_type

    def test_compare_traces(self):
        """Test trace cost comparison."""
        trace_a = Trace(name="test")
        span_a = Span(name="llm", model="gpt-4")
        span_a.token_usage = TokenUsage(total_tokens=100, total_cost=0.01)
        span_a.finish()
        trace_a.add_span(span_a)
        trace_a.finish()

        trace_b = Trace(name="test")
        span_b = Span(name="llm", model="gpt-4")
        span_b.token_usage = TokenUsage(total_tokens=200, total_cost=0.02)
        span_b.finish()
        trace_b.add_span(span_b)
        trace_b.finish()

        engine = CostDiffEngine()
        comparison = engine.compare(trace_a, trace_b)

        assert comparison.token_delta == 100
        assert comparison.cost_delta == pytest.approx(0.01)
        assert comparison.cost_change_percentage == pytest.approx(100.0)

    def test_most_expensive_spans(self):
        """Test finding most expensive spans."""
        trace = Trace(name="test")

        for i, cost in enumerate([0.01, 0.05, 0.02, 0.03]):
            span = Span(name=f"span-{i}")
            span.token_usage = TokenUsage(total_cost=cost)
            span.finish()
            trace.add_span(span)

        trace.finish()

        engine = CostDiffEngine()
        top = engine.get_most_expensive_spans(trace, top_k=2)

        assert len(top) == 2
        assert top[0].total_cost == 0.05
        assert top[1].total_cost == 0.03
