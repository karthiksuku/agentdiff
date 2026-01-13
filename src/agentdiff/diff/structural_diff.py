"""
Structural diff engine for comparing agent execution graphs.

Uses tree-matching algorithms to align execution graphs and identify
divergence points between agent runs.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from ..core.span import Span, SpanType
from ..core.trace import Trace


class DiffType(Enum):
    """Types of differences between spans."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"
    REORDERED = "reordered"


@dataclass
class SpanDiff:
    """Represents the diff between two spans."""
    diff_type: DiffType
    span_a: Optional[Span] = None
    span_b: Optional[Span] = None

    # Detailed changes
    field_changes: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)

    # Metrics
    token_delta: int = 0
    cost_delta: float = 0.0
    latency_delta_ms: float = 0.0
    confidence_delta: float = 0.0

    # Flags
    is_divergence_point: bool = False
    has_output_change: bool = False
    has_tool_change: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "diff_type": self.diff_type.value,
            "span_a_id": self.span_a.span_id if self.span_a else None,
            "span_b_id": self.span_b.span_id if self.span_b else None,
            "span_a_name": self.span_a.name if self.span_a else None,
            "span_b_name": self.span_b.name if self.span_b else None,
            "field_changes": {
                k: {"old": v[0], "new": v[1]}
                for k, v in self.field_changes.items()
            },
            "token_delta": self.token_delta,
            "cost_delta": self.cost_delta,
            "latency_delta_ms": self.latency_delta_ms,
            "confidence_delta": self.confidence_delta,
            "is_divergence_point": self.is_divergence_point,
            "has_output_change": self.has_output_change,
            "has_tool_change": self.has_tool_change,
        }


@dataclass
class TraceDiff:
    """Complete diff between two traces."""
    trace_a: Trace
    trace_b: Trace

    # Summary metrics
    structural_similarity: float = 0.0
    total_divergences: int = 0

    token_delta: int = 0
    cost_delta: float = 0.0
    latency_delta_ms: float = 0.0

    # Detailed diffs
    span_diffs: List[SpanDiff] = field(default_factory=list)

    # Key findings
    divergence_points: List[SpanDiff] = field(default_factory=list)
    regressions: List[SpanDiff] = field(default_factory=list)
    improvements: List[SpanDiff] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_a_id": self.trace_a.trace_id,
            "trace_b_id": self.trace_b.trace_id,
            "trace_a_name": self.trace_a.name,
            "trace_b_name": self.trace_b.name,
            "structural_similarity": self.structural_similarity,
            "total_divergences": self.total_divergences,
            "token_delta": self.token_delta,
            "cost_delta": self.cost_delta,
            "latency_delta_ms": self.latency_delta_ms,
            "span_diffs": [d.to_dict() for d in self.span_diffs],
            "divergence_points": [d.to_dict() for d in self.divergence_points],
            "regressions": [d.to_dict() for d in self.regressions],
            "improvements": [d.to_dict() for d in self.improvements],
            "summary": {
                "added_spans": len([d for d in self.span_diffs if d.diff_type == DiffType.ADDED]),
                "removed_spans": len([d for d in self.span_diffs if d.diff_type == DiffType.REMOVED]),
                "modified_spans": len([d for d in self.span_diffs if d.diff_type == DiffType.MODIFIED]),
                "unchanged_spans": len([d for d in self.span_diffs if d.diff_type == DiffType.UNCHANGED]),
            },
        }


class StructuralDiffEngine:
    """
    Engine for computing structural diffs between agent traces.

    Uses tree-matching algorithms to align execution graphs
    and identify divergence points.
    """

    def __init__(
        self,
        match_threshold: float = 0.7,
        weight_tokens: float = 0.3,
        weight_structure: float = 0.5,
        weight_output: float = 0.2,
    ):
        """
        Initialize the diff engine.

        Args:
            match_threshold: Minimum similarity for span matching
            weight_tokens: Weight for token similarity in scoring
            weight_structure: Weight for structural similarity
            weight_output: Weight for output similarity
        """
        self.match_threshold = match_threshold
        self.weight_tokens = weight_tokens
        self.weight_structure = weight_structure
        self.weight_output = weight_output

    def diff(self, trace_a: Trace, trace_b: Trace) -> TraceDiff:
        """
        Compute structural diff between two traces.

        Args:
            trace_a: First trace (baseline)
            trace_b: Second trace (comparison)

        Returns:
            TraceDiff with detailed comparison
        """
        result = TraceDiff(trace_a=trace_a, trace_b=trace_b)

        # Match spans between traces
        matched_pairs, unmatched_a, unmatched_b = self._match_spans(
            trace_a.spans, trace_b.spans
        )

        # Process matched pairs
        for span_a, span_b in matched_pairs:
            span_diff = self._diff_spans(span_a, span_b)
            result.span_diffs.append(span_diff)

            if span_diff.is_divergence_point:
                result.divergence_points.append(span_diff)
                result.total_divergences += 1

        # Process unmatched spans (added/removed)
        for span in unmatched_a:
            span_diff = SpanDiff(
                diff_type=DiffType.REMOVED,
                span_a=span,
                token_delta=-(span.token_usage.total_tokens if span.token_usage else 0),
                cost_delta=-(span.token_usage.total_cost if span.token_usage else 0),
            )
            result.span_diffs.append(span_diff)

        for span in unmatched_b:
            span_diff = SpanDiff(
                diff_type=DiffType.ADDED,
                span_b=span,
                token_delta=span.token_usage.total_tokens if span.token_usage else 0,
                cost_delta=span.token_usage.total_cost if span.token_usage else 0,
            )
            result.span_diffs.append(span_diff)

        # Calculate summary metrics
        result.structural_similarity = self._calculate_similarity(
            matched_pairs, unmatched_a, unmatched_b
        )

        result.token_delta = trace_b.total_tokens - trace_a.total_tokens
        result.cost_delta = trace_b.total_cost - trace_a.total_cost
        result.latency_delta_ms = trace_b.total_duration_ms - trace_a.total_duration_ms

        # Identify regressions and improvements
        self._classify_changes(result)

        return result

    def _match_spans(
        self,
        spans_a: List[Span],
        spans_b: List[Span],
    ) -> Tuple[List[Tuple[Span, Span]], List[Span], List[Span]]:
        """
        Match spans between two traces using multiple heuristics.

        Uses:
        1. Same name and type (exact match)
        2. Similar position in tree
        3. Similar input/output content

        Args:
            spans_a: Spans from first trace
            spans_b: Spans from second trace

        Returns:
            Tuple of (matched pairs, unmatched from A, unmatched from B)
        """
        matched: List[Tuple[Span, Span]] = []
        unmatched_a = list(spans_a)
        unmatched_b = list(spans_b)

        # First pass: exact name + type match
        for span_a in list(unmatched_a):
            for span_b in list(unmatched_b):
                if span_a.name == span_b.name and span_a.span_type == span_b.span_type:
                    matched.append((span_a, span_b))
                    unmatched_a.remove(span_a)
                    unmatched_b.remove(span_b)
                    break

        # Second pass: fuzzy matching for remaining
        for span_a in list(unmatched_a):
            best_match: Optional[Span] = None
            best_score = 0.0

            for span_b in unmatched_b:
                score = self._span_similarity(span_a, span_b)
                if score > self.match_threshold and score > best_score:
                    best_match = span_b
                    best_score = score

            if best_match:
                matched.append((span_a, best_match))
                unmatched_a.remove(span_a)
                unmatched_b.remove(best_match)

        return matched, unmatched_a, unmatched_b

    def _span_similarity(self, span_a: Span, span_b: Span) -> float:
        """
        Calculate similarity score between two spans.

        Args:
            span_a: First span
            span_b: Second span

        Returns:
            Similarity score (0-1)
        """
        score = 0.0

        # Name similarity
        if span_a.name == span_b.name:
            score += 0.4
        elif span_a.name and span_b.name:
            if span_a.name in span_b.name or span_b.name in span_a.name:
                score += 0.2

        # Type match
        if span_a.span_type == span_b.span_type:
            score += 0.3

        # Model match
        if span_a.model and span_b.model and span_a.model == span_b.model:
            score += 0.2

        # Tool name match (for tool calls)
        if span_a.span_type == SpanType.TOOL_CALL:
            tool_a = span_a.input_data.get("tool", "")
            tool_b = span_b.input_data.get("tool", "")
            if tool_a and tool_b and tool_a == tool_b:
                score += 0.1

        return min(score, 1.0)

    def _diff_spans(self, span_a: Span, span_b: Span) -> SpanDiff:
        """
        Create detailed diff between two matched spans.

        Args:
            span_a: First span
            span_b: Second span

        Returns:
            SpanDiff with detailed comparison
        """
        field_changes: Dict[str, Tuple[Any, Any]] = {}

        # Check for changes in key fields
        if span_a.model != span_b.model:
            field_changes["model"] = (span_a.model, span_b.model)

        if span_a.input_data != span_b.input_data:
            field_changes["input_data"] = (span_a.input_data, span_b.input_data)

        if span_a.output_data != span_b.output_data:
            field_changes["output_data"] = (span_a.output_data, span_b.output_data)

        if span_a.reasoning != span_b.reasoning:
            field_changes["reasoning"] = (span_a.reasoning, span_b.reasoning)

        # Calculate deltas
        token_delta = 0
        if span_a.token_usage and span_b.token_usage:
            token_delta = span_b.token_usage.total_tokens - span_a.token_usage.total_tokens

        cost_delta = 0.0
        if span_a.token_usage and span_b.token_usage:
            cost_a = span_a.token_usage.total_cost or 0
            cost_b = span_b.token_usage.total_cost or 0
            cost_delta = cost_b - cost_a

        latency_delta = (span_b.duration_ms or 0) - (span_a.duration_ms or 0)

        confidence_delta = 0.0
        if span_a.confidence_score is not None and span_b.confidence_score is not None:
            confidence_delta = span_b.confidence_score - span_a.confidence_score

        # Determine diff type
        if not field_changes:
            diff_type = DiffType.UNCHANGED
        else:
            diff_type = DiffType.MODIFIED

        # Check for divergence (different tool/decision)
        is_divergence = False
        has_tool_change = False

        if span_a.span_type == SpanType.TOOL_CALL:
            tool_a = span_a.input_data.get("tool", "")
            tool_b = span_b.input_data.get("tool", "")
            if tool_a != tool_b:
                is_divergence = True
                has_tool_change = True

        if span_a.reasoning and span_b.reasoning and span_a.reasoning != span_b.reasoning:
            is_divergence = True

        return SpanDiff(
            diff_type=diff_type,
            span_a=span_a,
            span_b=span_b,
            field_changes=field_changes,
            token_delta=token_delta,
            cost_delta=cost_delta,
            latency_delta_ms=latency_delta,
            confidence_delta=confidence_delta,
            is_divergence_point=is_divergence,
            has_output_change="output_data" in field_changes,
            has_tool_change=has_tool_change,
        )

    def _calculate_similarity(
        self,
        matched: List[Tuple[Span, Span]],
        unmatched_a: List[Span],
        unmatched_b: List[Span],
    ) -> float:
        """
        Calculate overall structural similarity.

        Args:
            matched: List of matched span pairs
            unmatched_a: Unmatched spans from trace A
            unmatched_b: Unmatched spans from trace B

        Returns:
            Similarity score (0-1)
        """
        total = len(matched) + len(unmatched_a) + len(unmatched_b)
        if total == 0:
            return 1.0

        # Count unchanged matches
        unchanged = sum(
            1 for a, b in matched
            if a.input_data == b.input_data and a.output_data == b.output_data
        )

        similarity = (2 * len(matched) + unchanged) / (2 * total)
        return min(similarity, 1.0)

    def _classify_changes(self, result: TraceDiff) -> None:
        """
        Classify changes as regressions or improvements.

        Args:
            result: TraceDiff to update with classifications
        """
        for diff in result.span_diffs:
            if diff.diff_type == DiffType.UNCHANGED:
                continue

            # Regression indicators
            is_regression = False
            if diff.confidence_delta < -0.1:  # Confidence dropped
                is_regression = True
            if diff.cost_delta > 0.01:  # Cost increased significantly
                is_regression = True
            if diff.latency_delta_ms > 1000:  # Latency increased > 1s
                is_regression = True

            # Improvement indicators
            is_improvement = False
            if diff.confidence_delta > 0.1:  # Confidence improved
                is_improvement = True
            if diff.cost_delta < -0.01:  # Cost decreased
                is_improvement = True
            if diff.latency_delta_ms < -500:  # Latency improved > 500ms
                is_improvement = True

            if is_regression and not is_improvement:
                result.regressions.append(diff)
            elif is_improvement and not is_regression:
                result.improvements.append(diff)
