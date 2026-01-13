"""
Diff renderer for displaying trace comparisons.

Supports multiple output formats including terminal, HTML, and Mermaid diagrams.
"""

from typing import Optional, List
from io import StringIO

from .structural_diff import TraceDiff, SpanDiff, DiffType
from .cost_diff import CostComparison


class DiffRenderer:
    """
    Renders trace diffs in various formats.
    """

    def __init__(self, color: bool = True):
        """
        Initialize the renderer.

        Args:
            color: Whether to use colored output (terminal)
        """
        self.color = color

    def render_terminal(self, diff: TraceDiff) -> str:
        """
        Render diff for terminal output.

        Args:
            diff: The TraceDiff to render

        Returns:
            Formatted string for terminal display
        """
        output = StringIO()

        # Header
        output.write("=" * 60 + "\n")
        output.write("AGENT DIFF REPORT\n")
        output.write("=" * 60 + "\n\n")

        # Trace info
        output.write(f"Trace A: {diff.trace_a.name} (v{diff.trace_a.version})\n")
        output.write(f"         ID: {diff.trace_a.trace_id[:8]}...\n")
        output.write(f"Trace B: {diff.trace_b.name} (v{diff.trace_b.version})\n")
        output.write(f"         ID: {diff.trace_b.trace_id[:8]}...\n\n")

        # Summary
        output.write("-" * 40 + "\n")
        output.write("SUMMARY\n")
        output.write("-" * 40 + "\n")
        output.write(f"Structural Similarity: {diff.structural_similarity:.1%}\n")
        output.write(f"Total Divergences: {diff.total_divergences}\n")
        output.write(f"Token Delta: {diff.token_delta:+d}\n")
        output.write(f"Cost Delta: ${diff.cost_delta:+.4f}\n")
        output.write(f"Latency Delta: {diff.latency_delta_ms:+.0f}ms\n\n")

        # Span summary
        added = len([d for d in diff.span_diffs if d.diff_type == DiffType.ADDED])
        removed = len([d for d in diff.span_diffs if d.diff_type == DiffType.REMOVED])
        modified = len([d for d in diff.span_diffs if d.diff_type == DiffType.MODIFIED])
        unchanged = len([d for d in diff.span_diffs if d.diff_type == DiffType.UNCHANGED])

        output.write(f"Spans: {self._color(f'+{added}', 'green')} added, ")
        output.write(f"{self._color(f'-{removed}', 'red')} removed, ")
        output.write(f"{self._color(f'~{modified}', 'yellow')} modified, ")
        output.write(f"{unchanged} unchanged\n\n")

        # Divergence points
        if diff.divergence_points:
            output.write("-" * 40 + "\n")
            output.write("DIVERGENCE POINTS\n")
            output.write("-" * 40 + "\n")
            for dp in diff.divergence_points:
                name = dp.span_a.name if dp.span_a else dp.span_b.name if dp.span_b else "unknown"
                output.write(f"  ! {name}\n")
                if dp.has_tool_change:
                    output.write("    Tool selection changed\n")
                if dp.has_output_change:
                    output.write("    Output changed\n")
            output.write("\n")

        # Regressions
        if diff.regressions:
            output.write("-" * 40 + "\n")
            output.write(self._color("REGRESSIONS", "red") + "\n")
            output.write("-" * 40 + "\n")
            for reg in diff.regressions:
                name = reg.span_a.name if reg.span_a else "unknown"
                output.write(f"  - {name}\n")
                if reg.cost_delta > 0:
                    output.write(f"    Cost: +${reg.cost_delta:.4f}\n")
                if reg.latency_delta_ms > 0:
                    output.write(f"    Latency: +{reg.latency_delta_ms:.0f}ms\n")
                if reg.confidence_delta < 0:
                    output.write(f"    Confidence: {reg.confidence_delta:+.2f}\n")
            output.write("\n")

        # Improvements
        if diff.improvements:
            output.write("-" * 40 + "\n")
            output.write(self._color("IMPROVEMENTS", "green") + "\n")
            output.write("-" * 40 + "\n")
            for imp in diff.improvements:
                name = imp.span_a.name if imp.span_a else "unknown"
                output.write(f"  + {name}\n")
                if imp.cost_delta < 0:
                    output.write(f"    Cost: ${imp.cost_delta:.4f}\n")
                if imp.latency_delta_ms < 0:
                    output.write(f"    Latency: {imp.latency_delta_ms:.0f}ms\n")
                if imp.confidence_delta > 0:
                    output.write(f"    Confidence: +{imp.confidence_delta:.2f}\n")
            output.write("\n")

        # Detailed changes
        output.write("-" * 40 + "\n")
        output.write("DETAILED CHANGES\n")
        output.write("-" * 40 + "\n")

        for span_diff in diff.span_diffs:
            self._render_span_diff(output, span_diff)

        output.write("=" * 60 + "\n")

        return output.getvalue()

    def _render_span_diff(self, output: StringIO, span_diff: SpanDiff) -> None:
        """Render a single span diff."""
        if span_diff.diff_type == DiffType.UNCHANGED:
            return

        prefix = {
            DiffType.ADDED: self._color("+", "green"),
            DiffType.REMOVED: self._color("-", "red"),
            DiffType.MODIFIED: self._color("~", "yellow"),
            DiffType.REORDERED: self._color("↔", "blue"),
        }.get(span_diff.diff_type, " ")

        name = (
            span_diff.span_a.name if span_diff.span_a
            else span_diff.span_b.name if span_diff.span_b
            else "unknown"
        )

        output.write(f"\n{prefix} {name}\n")

        if span_diff.field_changes:
            for field, (old, new) in span_diff.field_changes.items():
                if field in ("input_data", "output_data"):
                    output.write(f"    {field}: [changed]\n")
                else:
                    output.write(f"    {field}: {old} -> {new}\n")

        if span_diff.token_delta != 0:
            output.write(f"    tokens: {span_diff.token_delta:+d}\n")
        if span_diff.cost_delta != 0:
            output.write(f"    cost: ${span_diff.cost_delta:+.4f}\n")

    def _color(self, text: str, color: str) -> str:
        """Apply ANSI color to text."""
        if not self.color:
            return text

        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "reset": "\033[0m",
        }

        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def render_html(self, diff: TraceDiff) -> str:
        """
        Render diff as HTML.

        Args:
            diff: The TraceDiff to render

        Returns:
            HTML string
        """
        html = StringIO()

        html.write("""<!DOCTYPE html>
<html>
<head>
    <title>AgentDiff Report</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }
        .header { background: #1a1a2e; color: white; padding: 20px; border-radius: 8px; }
        .summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }
        .metric { background: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center; }
        .metric .value { font-size: 24px; font-weight: bold; }
        .metric .label { color: #666; font-size: 12px; }
        .section { margin: 30px 0; }
        .section-title { font-size: 18px; font-weight: bold; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .span-diff { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .added { background: #e6ffe6; border-left: 4px solid #00cc00; }
        .removed { background: #ffe6e6; border-left: 4px solid #cc0000; }
        .modified { background: #fff9e6; border-left: 4px solid #cc9900; }
        .divergence { background: #ffe6f0; border-left: 4px solid #cc0066; }
        .regression { color: #cc0000; }
        .improvement { color: #00cc00; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #eee; }
    </style>
</head>
<body>
""")

        # Header
        html.write(f"""
<div class="header">
    <h1>AgentDiff Report</h1>
    <p>Comparing {diff.trace_a.name} (v{diff.trace_a.version}) vs {diff.trace_b.name} (v{diff.trace_b.version})</p>
</div>
""")

        # Summary metrics
        html.write("""<div class="summary">""")
        html.write(f"""
    <div class="metric">
        <div class="value">{diff.structural_similarity:.0%}</div>
        <div class="label">Structural Similarity</div>
    </div>
    <div class="metric">
        <div class="value">{diff.total_divergences}</div>
        <div class="label">Divergences</div>
    </div>
    <div class="metric">
        <div class="value">{diff.token_delta:+d}</div>
        <div class="label">Token Delta</div>
    </div>
    <div class="metric">
        <div class="value">${diff.cost_delta:+.4f}</div>
        <div class="label">Cost Delta</div>
    </div>
""")
        html.write("""</div>""")

        # Divergence points
        if diff.divergence_points:
            html.write("""<div class="section">""")
            html.write("""<div class="section-title">Divergence Points</div>""")
            for dp in diff.divergence_points:
                name = dp.span_a.name if dp.span_a else dp.span_b.name if dp.span_b else "unknown"
                html.write(f"""<div class="span-diff divergence"><strong>{name}</strong>""")
                if dp.has_tool_change:
                    html.write(" - Tool selection changed")
                html.write("</div>")
            html.write("</div>")

        # Changes table
        html.write("""<div class="section">""")
        html.write("""<div class="section-title">All Changes</div>""")
        html.write("""<table><thead><tr>
            <th>Status</th><th>Span</th><th>Tokens</th><th>Cost</th><th>Latency</th>
        </tr></thead><tbody>""")

        for sd in diff.span_diffs:
            if sd.diff_type == DiffType.UNCHANGED:
                continue
            name = sd.span_a.name if sd.span_a else sd.span_b.name if sd.span_b else "unknown"
            status_class = sd.diff_type.value
            html.write(f"""<tr class="{status_class}">
                <td>{sd.diff_type.value}</td>
                <td>{name}</td>
                <td>{sd.token_delta:+d}</td>
                <td>${sd.cost_delta:+.4f}</td>
                <td>{sd.latency_delta_ms:+.0f}ms</td>
            </tr>""")

        html.write("</tbody></table></div>")

        html.write("""
</body>
</html>
""")

        return html.getvalue()

    def render_mermaid(self, diff: TraceDiff) -> str:
        """
        Render diff as Mermaid diagram.

        Args:
            diff: The TraceDiff to render

        Returns:
            Mermaid diagram string
        """
        mermaid = StringIO()

        mermaid.write("```mermaid\n")
        mermaid.write("graph TD\n")
        mermaid.write("    subgraph \"Trace A\"\n")

        # Render trace A spans
        for i, span in enumerate(diff.trace_a.spans):
            node_id = f"A{i}"
            label = span.name[:20] if span.name else f"span_{i}"
            mermaid.write(f'        {node_id}["{label}"]\n')

            if span.parent_span_id:
                # Find parent index
                for j, parent in enumerate(diff.trace_a.spans):
                    if parent.span_id == span.parent_span_id:
                        mermaid.write(f"        A{j} --> {node_id}\n")
                        break

        mermaid.write("    end\n\n")
        mermaid.write("    subgraph \"Trace B\"\n")

        # Render trace B spans
        for i, span in enumerate(diff.trace_b.spans):
            node_id = f"B{i}"
            label = span.name[:20] if span.name else f"span_{i}"

            # Check if this span is modified/added
            style = ""
            for sd in diff.span_diffs:
                if sd.span_b and sd.span_b.span_id == span.span_id:
                    if sd.diff_type == DiffType.ADDED:
                        style = ":::added"
                    elif sd.diff_type == DiffType.MODIFIED:
                        style = ":::modified"
                    break

            mermaid.write(f'        {node_id}["{label}"]{style}\n')

            if span.parent_span_id:
                for j, parent in enumerate(diff.trace_b.spans):
                    if parent.span_id == span.parent_span_id:
                        mermaid.write(f"        B{j} --> {node_id}\n")
                        break

        mermaid.write("    end\n\n")

        # Add matching lines
        for sd in diff.span_diffs:
            if sd.span_a and sd.span_b:
                idx_a = next(
                    (i for i, s in enumerate(diff.trace_a.spans) if s.span_id == sd.span_a.span_id),
                    None
                )
                idx_b = next(
                    (i for i, s in enumerate(diff.trace_b.spans) if s.span_id == sd.span_b.span_id),
                    None
                )
                if idx_a is not None and idx_b is not None:
                    style = "-..->" if sd.diff_type == DiffType.MODIFIED else "==>"
                    mermaid.write(f"    A{idx_a} {style} B{idx_b}\n")

        # Add styles
        mermaid.write("\n    classDef added fill:#90EE90\n")
        mermaid.write("    classDef modified fill:#FFE4B5\n")
        mermaid.write("    classDef removed fill:#FFB6C1\n")

        mermaid.write("```\n")

        return mermaid.getvalue()

    def render_cost_comparison(self, comparison: CostComparison) -> str:
        """
        Render cost comparison for terminal output.

        Args:
            comparison: The CostComparison to render

        Returns:
            Formatted string
        """
        output = StringIO()

        output.write("=" * 60 + "\n")
        output.write("COST COMPARISON REPORT\n")
        output.write("=" * 60 + "\n\n")

        # Overall comparison
        output.write(f"Trace A: {comparison.trace_a.trace_name}\n")
        output.write(f"  Total Cost: ${comparison.trace_a.total_cost:.4f}\n")
        output.write(f"  Total Tokens: {comparison.trace_a.total_tokens:,}\n\n")

        output.write(f"Trace B: {comparison.trace_b.trace_name}\n")
        output.write(f"  Total Cost: ${comparison.trace_b.total_cost:.4f}\n")
        output.write(f"  Total Tokens: {comparison.trace_b.total_tokens:,}\n\n")

        # Delta
        output.write("-" * 40 + "\n")
        delta_color = "red" if comparison.cost_delta > 0 else "green"
        output.write(f"Cost Delta: {self._color(f'${comparison.cost_delta:+.4f}', delta_color)}\n")
        output.write(f"Change: {comparison.cost_change_percentage:+.1f}%\n\n")

        # By model
        if comparison.model_cost_changes:
            output.write("-" * 40 + "\n")
            output.write("COST BY MODEL\n")
            output.write("-" * 40 + "\n")
            for model, changes in comparison.model_cost_changes.items():
                delta = changes["delta"]
                color = "red" if delta > 0 else "green" if delta < 0 else "reset"
                output.write(f"  {model}:\n")
                output.write(f"    Before: ${changes['before']:.4f}\n")
                output.write(f"    After:  ${changes['after']:.4f}\n")
                output.write(f"    Delta:  {self._color(f'${delta:+.4f}', color)}\n")
            output.write("\n")

        # Suggestions
        if comparison.optimization_suggestions:
            output.write("-" * 40 + "\n")
            output.write("OPTIMIZATION SUGGESTIONS\n")
            output.write("-" * 40 + "\n")
            for suggestion in comparison.optimization_suggestions:
                output.write(f"  * {suggestion}\n")
            output.write("\n")

        output.write("=" * 60 + "\n")

        return output.getvalue()
