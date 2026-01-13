"""
HTML export for trace visualization.
"""

from typing import Optional
from io import StringIO
import html

from ..core.trace import Trace
from ..core.span import SpanType, SpanStatus


class HTMLExporter:
    """
    Exports traces to HTML for web viewing.
    """

    def __init__(self):
        """Initialize the exporter."""
        pass

    def export_trace(self, trace: Trace, title: Optional[str] = None) -> str:
        """
        Export a trace to HTML.

        Args:
            trace: The trace to export
            title: Optional custom title

        Returns:
            HTML string
        """
        title = title or f"Trace: {trace.name}"

        html_content = StringIO()

        # HTML header
        html_content.write(self._get_html_header(title))

        # Trace header
        html_content.write(f"""
<div class="container">
    <div class="trace-header">
        <h1>{html.escape(trace.name)}</h1>
        <div class="trace-meta">
            <span class="badge">{trace.version}</span>
            <span class="badge">{trace.branch}</span>
            <span class="badge status-{trace.status}">{trace.status}</span>
        </div>
    </div>

    <div class="metrics">
        <div class="metric">
            <div class="metric-value">{trace.total_duration_ms:.0f}ms</div>
            <div class="metric-label">Duration</div>
        </div>
        <div class="metric">
            <div class="metric-value">{trace.total_tokens:,}</div>
            <div class="metric-label">Tokens</div>
        </div>
        <div class="metric">
            <div class="metric-value">${trace.total_cost:.4f}</div>
            <div class="metric-label">Cost</div>
        </div>
        <div class="metric">
            <div class="metric-value">{len(trace.spans)}</div>
            <div class="metric-label">Spans</div>
        </div>
    </div>
""")

        # Span timeline
        html_content.write("""
    <h2>Execution Timeline</h2>
    <div class="timeline">
""")

        for span in trace.spans:
            html_content.write(self._render_span_card(span))

        html_content.write("""
    </div>
</div>
""")

        # HTML footer
        html_content.write(self._get_html_footer())

        return html_content.getvalue()

    def _render_span_card(self, span) -> str:
        """Render a span as an HTML card."""
        icon = self._get_span_icon(span.span_type)
        status_class = f"status-{span.status.value}"

        tokens_info = ""
        if span.token_usage:
            tokens_info = f"""
            <div class="span-tokens">
                {span.token_usage.total_tokens:,} tokens
                (${span.token_usage.total_cost:.4f})
            </div>
"""

        error_info = ""
        if span.error:
            error_info = f"""
            <div class="span-error">
                Error: {html.escape(span.error[:100])}
            </div>
"""

        return f"""
        <div class="span-card {status_class}">
            <div class="span-header">
                <span class="span-icon">{icon}</span>
                <span class="span-name">{html.escape(span.name or 'unnamed')}</span>
                <span class="span-type">{span.span_type.value}</span>
            </div>
            <div class="span-meta">
                {f'<span>Model: {span.model}</span>' if span.model else ''}
                {f'<span>Duration: {span.duration_ms:.0f}ms</span>' if span.duration_ms else ''}
            </div>
            {tokens_info}
            {error_info}
        </div>
"""

    def _get_span_icon(self, span_type: SpanType) -> str:
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

    def _get_html_header(self, title: str) -> str:
        """Get HTML header with styles."""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --text-primary: #c9d1d9;
            --text-secondary: #8b949e;
            --accent-blue: #58a6ff;
            --accent-green: #3fb950;
            --accent-red: #f85149;
            --accent-yellow: #d29922;
            --accent-purple: #a371f7;
            --border-color: #30363d;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }}

        .trace-header {{
            margin-bottom: 2rem;
        }}

        .trace-header h1 {{
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }}

        .trace-meta {{
            display: flex;
            gap: 0.5rem;
        }}

        .badge {{
            background: var(--bg-tertiary);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
        }}

        .status-completed {{
            background: var(--accent-green);
            color: white;
        }}

        .status-failed {{
            background: var(--accent-red);
            color: white;
        }}

        .status-running {{
            background: var(--accent-yellow);
            color: black;
        }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .metric {{
            background: var(--bg-secondary);
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
            border: 1px solid var(--border-color);
        }}

        .metric-value {{
            font-size: 1.5rem;
            font-weight: bold;
            color: var(--accent-blue);
        }}

        .metric-label {{
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}

        h2 {{
            margin: 2rem 0 1rem;
            font-size: 1.5rem;
        }}

        .timeline {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}

        .span-card {{
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            border-left: 4px solid var(--accent-blue);
        }}

        .span-card.status-failed {{
            border-left-color: var(--accent-red);
        }}

        .span-card.status-completed {{
            border-left-color: var(--accent-green);
        }}

        .span-header {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.5rem;
        }}

        .span-icon {{
            font-size: 1.25rem;
        }}

        .span-name {{
            font-weight: 600;
            flex-grow: 1;
        }}

        .span-type {{
            color: var(--text-secondary);
            font-size: 0.875rem;
            background: var(--bg-tertiary);
            padding: 0.125rem 0.5rem;
            border-radius: 4px;
        }}

        .span-meta {{
            color: var(--text-secondary);
            font-size: 0.875rem;
            display: flex;
            gap: 1rem;
        }}

        .span-tokens {{
            margin-top: 0.5rem;
            color: var(--accent-purple);
        }}

        .span-error {{
            margin-top: 0.5rem;
            color: var(--accent-red);
            font-size: 0.875rem;
        }}

        footer {{
            margin-top: 3rem;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }}

        footer a {{
            color: var(--accent-blue);
            text-decoration: none;
        }}
    </style>
</head>
<body>
"""

    def _get_html_footer(self) -> str:
        """Get HTML footer."""
        return """
    <footer>
        <p>Generated by <a href="https://github.com/agentdiff/agentdiff">AgentDiff</a></p>
    </footer>
</body>
</html>
"""
