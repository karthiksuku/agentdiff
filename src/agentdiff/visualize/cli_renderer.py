"""
CLI renderer for terminal visualization.
"""

from typing import Optional
from io import StringIO

from ..core.trace import Trace
from ..core.span import SpanType, SpanStatus


class CLIRenderer:
    """
    Renders traces and diffs for terminal output.
    """

    def __init__(self, color: bool = True, width: int = 80):
        """
        Initialize the renderer.

        Args:
            color: Whether to use ANSI colors
            width: Terminal width
        """
        self.color = color
        self.width = width

    def _c(self, text: str, color: str) -> str:
        """Apply color to text."""
        if not self.color:
            return text

        colors = {
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "white": "\033[97m",
            "bold": "\033[1m",
            "dim": "\033[2m",
            "reset": "\033[0m",
        }

        return f"{colors.get(color, '')}{text}{colors['reset']}"

    def render_trace(self, trace: Trace) -> str:
        """
        Render a trace for terminal display.

        Args:
            trace: The trace to render

        Returns:
            Formatted string
        """
        output = StringIO()

        # Header
        output.write(self._c("=" * self.width, "dim") + "\n")
        output.write(self._c(f" TRACE: {trace.name}", "bold") + "\n")
        output.write(self._c("=" * self.width, "dim") + "\n\n")

        # Info
        output.write(f"  ID:       {trace.trace_id}\n")
        output.write(f"  Version:  {trace.version}\n")
        output.write(f"  Branch:   {trace.branch}\n")
        output.write(f"  Status:   {self._status_color(trace.status)}\n")
        output.write(f"  Duration: {trace.total_duration_ms:.0f}ms\n")
        output.write(f"  Tokens:   {trace.total_tokens:,}\n")
        output.write(f"  Cost:     ${trace.total_cost:.4f}\n")
        output.write(f"  Spans:    {len(trace.spans)}\n")
        output.write("\n")

        # Span tree
        output.write(self._c("-" * self.width, "dim") + "\n")
        output.write(self._c(" EXECUTION TREE", "bold") + "\n")
        output.write(self._c("-" * self.width, "dim") + "\n\n")

        self._render_span_tree(output, trace)

        output.write("\n" + self._c("=" * self.width, "dim") + "\n")

        return output.getvalue()

    def _status_color(self, status: str) -> str:
        """Get colored status string."""
        colors = {
            "running": "yellow",
            "completed": "green",
            "failed": "red",
            "cancelled": "dim",
        }
        return self._c(status, colors.get(status, "white"))

    def _span_icon(self, span_type: SpanType) -> str:
        """Get icon for span type."""
        icons = {
            SpanType.LLM_CALL: "🤖",
            SpanType.TOOL_CALL: "🔧",
            SpanType.REASONING: "💭",
            SpanType.RETRIEVAL: "📚",
            SpanType.PLANNING: "📋",
            SpanType.MEMORY_ACCESS: "💾",
            SpanType.AGENT_STEP: "👤",
            SpanType.EMBEDDING: "🔢",
            SpanType.CUSTOM: "•",
        }
        return icons.get(span_type, "•")

    def _render_span_tree(self, output: StringIO, trace: Trace) -> None:
        """Render span tree."""
        tree = trace.get_span_tree()

        def render_node(node, prefix="", is_last=True):
            if node["span"]:
                span = node["span"]
                connector = "└── " if is_last else "├── "
                icon = self._span_icon(span.span_type)

                # Format span info
                name = span.name[:30] if span.name else "unnamed"
                duration = f"{span.duration_ms:.0f}ms" if span.duration_ms else ""
                tokens = ""
                if span.token_usage:
                    tokens = f"{span.token_usage.total_tokens:,} tokens"

                # Color based on status
                status_color = {
                    SpanStatus.COMPLETED: "green",
                    SpanStatus.FAILED: "red",
                    SpanStatus.RUNNING: "yellow",
                    SpanStatus.CANCELLED: "dim",
                }.get(span.status, "white")

                line = f"{prefix}{connector}{icon} {self._c(name, status_color)}"
                if duration or tokens:
                    line += self._c(f" ({duration}, {tokens})", "dim")

                output.write(line + "\n")

                # Render children
                child_prefix = prefix + ("    " if is_last else "│   ")
                children = node.get("children", [])
                for i, child in enumerate(children):
                    render_node(child, child_prefix, i == len(children) - 1)

        if tree["span"]:
            render_node(tree, is_last=True)
        else:
            for i, child in enumerate(tree.get("children", [])):
                render_node(child, "", i == len(tree["children"]) - 1)

    def render_summary(self, trace: Trace) -> str:
        """
        Render a brief trace summary.

        Args:
            trace: The trace

        Returns:
            Summary string
        """
        status_icon = {"completed": "✓", "failed": "✗", "running": "⟳"}.get(trace.status, "?")
        status_color = {"completed": "green", "failed": "red", "running": "yellow"}.get(trace.status, "white")

        return (
            f"{self._c(status_icon, status_color)} {trace.name} "
            f"(v{trace.version}) - {len(trace.spans)} spans, "
            f"${trace.total_cost:.4f}, {trace.total_duration_ms:.0f}ms"
        )

    def render_span(self, span) -> str:
        """
        Render a single span.

        Args:
            span: The span to render

        Returns:
            Formatted string
        """
        output = StringIO()

        icon = self._span_icon(span.span_type)
        status_color = {
            SpanStatus.COMPLETED: "green",
            SpanStatus.FAILED: "red",
            SpanStatus.RUNNING: "yellow",
        }.get(span.status, "white")

        output.write(f"{icon} {self._c(span.name, 'bold')}\n")
        output.write(f"   Type:   {span.span_type.value}\n")
        output.write(f"   Status: {self._c(span.status.value, status_color)}\n")

        if span.model:
            output.write(f"   Model:  {span.model}\n")

        if span.duration_ms:
            output.write(f"   Time:   {span.duration_ms:.0f}ms\n")

        if span.token_usage:
            output.write(f"   Tokens: {span.token_usage.total_tokens:,}\n")
            output.write(f"   Cost:   ${span.token_usage.total_cost:.4f}\n")

        if span.error:
            output.write(f"   Error:  {self._c(span.error[:50], 'red')}\n")

        return output.getvalue()
