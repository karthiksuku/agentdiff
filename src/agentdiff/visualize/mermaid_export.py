"""
Mermaid diagram export for trace visualization.
"""

from typing import Optional
from io import StringIO

from ..core.trace import Trace
from ..core.span import SpanType


class MermaidExporter:
    """
    Exports traces to Mermaid diagrams.
    """

    def __init__(self):
        """Initialize the exporter."""
        pass

    def export_trace(
        self,
        trace: Trace,
        diagram_type: str = "flowchart",
        direction: str = "TD",
    ) -> str:
        """
        Export a trace to Mermaid diagram.

        Args:
            trace: The trace to export
            diagram_type: Type of diagram (flowchart, sequence)
            direction: Diagram direction (TD, LR, BT, RL)

        Returns:
            Mermaid diagram string
        """
        if diagram_type == "sequence":
            return self._export_sequence(trace)
        else:
            return self._export_flowchart(trace, direction)

    def _export_flowchart(self, trace: Trace, direction: str = "TD") -> str:
        """Export as flowchart."""
        output = StringIO()

        output.write(f"```mermaid\nflowchart {direction}\n")

        # Add title
        output.write(f"    subgraph {self._escape(trace.name)}\n")

        # Create nodes
        for i, span in enumerate(trace.spans):
            node_id = f"span_{i}"
            label = self._escape(span.name[:30] if span.name else f"span_{i}")
            shape = self._get_node_shape(span.span_type)

            output.write(f"        {node_id}{shape[0]}\"{label}\"{shape[1]}\n")

        # Create edges
        for i, span in enumerate(trace.spans):
            if span.parent_span_id:
                # Find parent index
                for j, parent in enumerate(trace.spans):
                    if parent.span_id == span.parent_span_id:
                        parent_id = f"span_{j}"
                        child_id = f"span_{i}"
                        output.write(f"        {parent_id} --> {child_id}\n")
                        break

        output.write("    end\n")

        # Add styles
        output.write("\n")
        for i, span in enumerate(trace.spans):
            style_class = self._get_style_class(span.span_type)
            output.write(f"    class span_{i} {style_class}\n")

        # Define classes
        output.write("""
    classDef llm fill:#4a9eff,stroke:#2171c7,color:#fff
    classDef tool fill:#3fb950,stroke:#2ea043,color:#fff
    classDef reasoning fill:#a371f7,stroke:#8957e5,color:#fff
    classDef retrieval fill:#d29922,stroke:#9e6a03,color:#fff
    classDef default fill:#6e7681,stroke:#484f58,color:#fff
""")

        output.write("```\n")

        return output.getvalue()

    def _export_sequence(self, trace: Trace) -> str:
        """Export as sequence diagram."""
        output = StringIO()

        output.write("```mermaid\nsequenceDiagram\n")
        output.write(f"    title {self._escape(trace.name)}\n")

        # Define participants
        participants = set()
        for span in trace.spans:
            if span.span_type == SpanType.LLM_CALL and span.model:
                participants.add(f"LLM_{span.model}")
            elif span.span_type == SpanType.TOOL_CALL:
                tool_name = span.input_data.get("tool", span.name)
                participants.add(f"Tool_{tool_name}")

        participants.add("Agent")

        for p in sorted(participants):
            clean_name = self._escape(p.replace("_", " "))
            output.write(f"    participant {p} as {clean_name}\n")

        output.write("\n")

        # Create sequence
        for span in trace.spans:
            if span.span_type == SpanType.LLM_CALL and span.model:
                target = f"LLM_{span.model}"
                output.write(f"    Agent->>+{target}: {self._escape(span.name[:20])}\n")
                if span.output_data.get("response"):
                    response = str(span.output_data["response"])[:30]
                    output.write(f"    {target}-->>-Agent: {self._escape(response)}\n")

            elif span.span_type == SpanType.TOOL_CALL:
                tool_name = span.input_data.get("tool", span.name)
                target = f"Tool_{tool_name}"
                output.write(f"    Agent->>+{target}: call\n")
                output.write(f"    {target}-->>-Agent: result\n")

        output.write("```\n")

        return output.getvalue()

    def _get_node_shape(self, span_type: SpanType) -> tuple:
        """Get node shape brackets for span type."""
        shapes = {
            SpanType.LLM_CALL: ("[/", "\\]"),  # Parallelogram
            SpanType.TOOL_CALL: ("[[", "]]"),  # Subroutine
            SpanType.REASONING: ("{{", "}}"),  # Hexagon
            SpanType.RETRIEVAL: ("[(", ")]"),  # Cylindrical
            SpanType.PLANNING: ("[", "]"),     # Rectangle
        }
        return shapes.get(span_type, ("[", "]"))

    def _get_style_class(self, span_type: SpanType) -> str:
        """Get style class for span type."""
        classes = {
            SpanType.LLM_CALL: "llm",
            SpanType.TOOL_CALL: "tool",
            SpanType.REASONING: "reasoning",
            SpanType.RETRIEVAL: "retrieval",
        }
        return classes.get(span_type, "default")

    def _escape(self, text: str) -> str:
        """Escape special characters for Mermaid."""
        if not text:
            return "unnamed"
        # Remove or replace problematic characters
        text = text.replace('"', "'")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace("{", "(")
        text = text.replace("}", ")")
        text = text.replace("[", "(")
        text = text.replace("]", ")")
        return text

    def export_diff(
        self,
        trace_a: Trace,
        trace_b: Trace,
    ) -> str:
        """
        Export a diff between two traces as Mermaid diagram.

        Args:
            trace_a: First trace
            trace_b: Second trace

        Returns:
            Mermaid diagram string
        """
        output = StringIO()

        output.write("```mermaid\nflowchart LR\n")

        # Trace A subgraph
        output.write(f"    subgraph A[\"{self._escape(trace_a.name)} v{trace_a.version}\"]\n")
        for i, span in enumerate(trace_a.spans):
            node_id = f"A{i}"
            label = self._escape(span.name[:20] if span.name else f"span_{i}")
            output.write(f"        {node_id}[\"{label}\"]\n")

            if span.parent_span_id:
                for j, parent in enumerate(trace_a.spans):
                    if parent.span_id == span.parent_span_id:
                        output.write(f"        A{j} --> {node_id}\n")
                        break

        output.write("    end\n\n")

        # Trace B subgraph
        output.write(f"    subgraph B[\"{self._escape(trace_b.name)} v{trace_b.version}\"]\n")
        for i, span in enumerate(trace_b.spans):
            node_id = f"B{i}"
            label = self._escape(span.name[:20] if span.name else f"span_{i}")
            output.write(f"        {node_id}[\"{label}\"]\n")

            if span.parent_span_id:
                for j, parent in enumerate(trace_b.spans):
                    if parent.span_id == span.parent_span_id:
                        output.write(f"        B{j} --> {node_id}\n")
                        break

        output.write("    end\n\n")

        # Connection between matching spans (simplified)
        output.write("    A0 -.-> B0\n")

        output.write("```\n")

        return output.getvalue()
