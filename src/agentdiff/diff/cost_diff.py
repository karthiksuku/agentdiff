"""
Cost diff engine for comparing costs between agent traces.

Provides detailed cost analysis and comparison between different
versions of agent executions.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..core.span import Span, SpanType
from ..core.trace import Trace
from ..core.cost_tracker import get_cost_tracker


@dataclass
class SpanCostBreakdown:
    """Cost breakdown for a single span."""
    span_id: str
    span_name: str
    span_type: SpanType

    model: Optional[str] = None
    provider: Optional[str] = None

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0

    # Percentage of trace total
    cost_percentage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "span_id": self.span_id,
            "span_name": self.span_name,
            "span_type": self.span_type.value,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
            "cost_percentage": self.cost_percentage,
        }


@dataclass
class TraceCostBreakdown:
    """Cost breakdown for an entire trace."""
    trace_id: str
    trace_name: str

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0

    total_input_cost: float = 0.0
    total_output_cost: float = 0.0
    total_cost: float = 0.0

    # Breakdown by span
    span_costs: List[SpanCostBreakdown] = field(default_factory=list)

    # Breakdown by model
    cost_by_model: Dict[str, float] = field(default_factory=dict)

    # Breakdown by span type
    cost_by_type: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "trace_name": self.trace_name,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_input_cost": self.total_input_cost,
            "total_output_cost": self.total_output_cost,
            "total_cost": self.total_cost,
            "span_costs": [s.to_dict() for s in self.span_costs],
            "cost_by_model": self.cost_by_model,
            "cost_by_type": self.cost_by_type,
        }


@dataclass
class CostComparison:
    """Comparison of costs between two traces."""
    trace_a: TraceCostBreakdown
    trace_b: TraceCostBreakdown

    # Deltas
    token_delta: int = 0
    cost_delta: float = 0.0
    cost_change_percentage: float = 0.0

    # Per-model changes
    model_cost_changes: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-type changes
    type_cost_changes: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Recommendations
    optimization_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_a": self.trace_a.to_dict(),
            "trace_b": self.trace_b.to_dict(),
            "token_delta": self.token_delta,
            "cost_delta": self.cost_delta,
            "cost_change_percentage": self.cost_change_percentage,
            "model_cost_changes": self.model_cost_changes,
            "type_cost_changes": self.type_cost_changes,
            "optimization_suggestions": self.optimization_suggestions,
        }


class CostDiffEngine:
    """
    Engine for computing cost diffs between agent traces.

    Provides detailed cost breakdowns and comparisons, along with
    optimization suggestions.
    """

    def __init__(self):
        """Initialize the cost diff engine."""
        self.cost_tracker = get_cost_tracker()

    def analyze_trace(self, trace: Trace) -> TraceCostBreakdown:
        """
        Analyze costs for a single trace.

        Args:
            trace: The trace to analyze

        Returns:
            TraceCostBreakdown with detailed cost information
        """
        breakdown = TraceCostBreakdown(
            trace_id=trace.trace_id,
            trace_name=trace.name,
        )

        for span in trace.spans:
            span_cost = self._analyze_span(span)
            breakdown.span_costs.append(span_cost)

            # Accumulate totals
            breakdown.total_input_tokens += span_cost.input_tokens
            breakdown.total_output_tokens += span_cost.output_tokens
            breakdown.total_tokens += span_cost.total_tokens
            breakdown.total_input_cost += span_cost.input_cost
            breakdown.total_output_cost += span_cost.output_cost
            breakdown.total_cost += span_cost.total_cost

            # Accumulate by model
            if span_cost.model:
                if span_cost.model not in breakdown.cost_by_model:
                    breakdown.cost_by_model[span_cost.model] = 0.0
                breakdown.cost_by_model[span_cost.model] += span_cost.total_cost

            # Accumulate by type
            type_name = span_cost.span_type.value
            if type_name not in breakdown.cost_by_type:
                breakdown.cost_by_type[type_name] = 0.0
            breakdown.cost_by_type[type_name] += span_cost.total_cost

        # Calculate percentages
        if breakdown.total_cost > 0:
            for span_cost in breakdown.span_costs:
                span_cost.cost_percentage = (
                    span_cost.total_cost / breakdown.total_cost
                ) * 100

        return breakdown

    def _analyze_span(self, span: Span) -> SpanCostBreakdown:
        """
        Analyze costs for a single span.

        Args:
            span: The span to analyze

        Returns:
            SpanCostBreakdown with cost information
        """
        breakdown = SpanCostBreakdown(
            span_id=span.span_id,
            span_name=span.name,
            span_type=span.span_type,
            model=span.model,
            provider=span.provider,
        )

        if span.token_usage:
            breakdown.input_tokens = span.token_usage.input_tokens
            breakdown.output_tokens = span.token_usage.output_tokens
            breakdown.total_tokens = span.token_usage.total_tokens
            breakdown.input_cost = span.token_usage.input_cost
            breakdown.output_cost = span.token_usage.output_cost
            breakdown.total_cost = span.token_usage.total_cost

        return breakdown

    def compare(self, trace_a: Trace, trace_b: Trace) -> CostComparison:
        """
        Compare costs between two traces.

        Args:
            trace_a: First trace (baseline)
            trace_b: Second trace (comparison)

        Returns:
            CostComparison with detailed analysis
        """
        breakdown_a = self.analyze_trace(trace_a)
        breakdown_b = self.analyze_trace(trace_b)

        comparison = CostComparison(
            trace_a=breakdown_a,
            trace_b=breakdown_b,
        )

        # Calculate deltas
        comparison.token_delta = breakdown_b.total_tokens - breakdown_a.total_tokens
        comparison.cost_delta = breakdown_b.total_cost - breakdown_a.total_cost

        if breakdown_a.total_cost > 0:
            comparison.cost_change_percentage = (
                comparison.cost_delta / breakdown_a.total_cost
            ) * 100

        # Calculate per-model changes
        all_models = set(breakdown_a.cost_by_model.keys()) | set(breakdown_b.cost_by_model.keys())
        for model in all_models:
            cost_a = breakdown_a.cost_by_model.get(model, 0.0)
            cost_b = breakdown_b.cost_by_model.get(model, 0.0)
            comparison.model_cost_changes[model] = {
                "before": cost_a,
                "after": cost_b,
                "delta": cost_b - cost_a,
            }

        # Calculate per-type changes
        all_types = set(breakdown_a.cost_by_type.keys()) | set(breakdown_b.cost_by_type.keys())
        for type_name in all_types:
            cost_a = breakdown_a.cost_by_type.get(type_name, 0.0)
            cost_b = breakdown_b.cost_by_type.get(type_name, 0.0)
            comparison.type_cost_changes[type_name] = {
                "before": cost_a,
                "after": cost_b,
                "delta": cost_b - cost_a,
            }

        # Generate optimization suggestions
        comparison.optimization_suggestions = self._generate_suggestions(comparison)

        return comparison

    def _generate_suggestions(self, comparison: CostComparison) -> List[str]:
        """
        Generate cost optimization suggestions.

        Args:
            comparison: The cost comparison

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Check for expensive models
        for model, costs in comparison.model_cost_changes.items():
            if costs["after"] > 0.1:  # More than $0.10
                if "gpt-4" in model.lower():
                    suggestions.append(
                        f"Consider using gpt-4o-mini instead of {model} for simpler tasks"
                    )
                elif "claude-3-opus" in model.lower():
                    suggestions.append(
                        f"Consider using claude-3-sonnet instead of {model} for simpler tasks"
                    )

        # Check output token ratio
        trace_b = comparison.trace_b
        if trace_b.total_tokens > 0:
            output_ratio = trace_b.total_output_tokens / trace_b.total_tokens
            if output_ratio > 0.7:
                suggestions.append(
                    "High output token ratio detected. Consider requesting more concise responses."
                )

        # Check for cost increases
        if comparison.cost_change_percentage > 50:
            suggestions.append(
                f"Cost increased by {comparison.cost_change_percentage:.1f}%. "
                "Review the changes for optimization opportunities."
            )

        # Check LLM call frequency
        llm_cost_a = comparison.trace_a.cost_by_type.get("llm_call", 0)
        llm_cost_b = comparison.trace_b.cost_by_type.get("llm_call", 0)
        if llm_cost_b > llm_cost_a * 1.5:
            suggestions.append(
                "LLM call costs increased significantly. Consider caching or reducing call frequency."
            )

        return suggestions

    def get_most_expensive_spans(
        self,
        trace: Trace,
        top_k: int = 5,
    ) -> List[SpanCostBreakdown]:
        """
        Get the most expensive spans in a trace.

        Args:
            trace: The trace to analyze
            top_k: Number of top spans to return

        Returns:
            List of most expensive span breakdowns
        """
        breakdown = self.analyze_trace(trace)
        sorted_spans = sorted(
            breakdown.span_costs,
            key=lambda x: x.total_cost,
            reverse=True,
        )
        return sorted_spans[:top_k]

    def estimate_cost_savings(
        self,
        trace: Trace,
        target_model: str,
    ) -> Dict[str, Any]:
        """
        Estimate cost savings by switching to a different model.

        Args:
            trace: The trace to analyze
            target_model: The target model to switch to

        Returns:
            Dictionary with savings analysis
        """
        breakdown = self.analyze_trace(trace)

        current_cost = breakdown.total_cost
        estimated_cost = 0.0

        target_pricing = self.cost_tracker.get_pricing(target_model)
        if not target_pricing:
            return {"error": f"Unknown model: {target_model}"}

        for span_cost in breakdown.span_costs:
            if span_cost.model:
                # Estimate cost with target model
                new_cost = target_pricing.calculate_cost(
                    span_cost.input_tokens,
                    span_cost.output_tokens,
                )
                estimated_cost += new_cost
            else:
                estimated_cost += span_cost.total_cost

        savings = current_cost - estimated_cost
        savings_percentage = (savings / current_cost * 100) if current_cost > 0 else 0

        return {
            "current_cost": current_cost,
            "estimated_cost": estimated_cost,
            "savings": savings,
            "savings_percentage": savings_percentage,
            "target_model": target_model,
        }
