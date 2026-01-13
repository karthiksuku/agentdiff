"""
AgentDiff CLI - Command line interface for agent diff and replay.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import click

from ..storage.sqlite_store import SQLiteStore
from ..diff.structural_diff import StructuralDiffEngine
from ..diff.semantic_diff import SemanticDiffEngine
from ..diff.cost_diff import CostDiffEngine
from ..diff.renderer import DiffRenderer
from ..replay.engine import ReplayEngine, ReplayMode


def run_async(coro):
    """Run an async coroutine."""
    return asyncio.get_event_loop().run_until_complete(coro)


@click.group()
@click.version_option(version="0.1.0", prog_name="agentdiff")
@click.option(
    "--db",
    default="agentdiff.db",
    help="Path to the database file",
    envvar="AGENTDIFF_DB",
)
@click.pass_context
def cli(ctx, db):
    """
    AgentDiff - Visual Diff & Replay Debugger for AI Agents

    Compare agent executions, identify divergences, and replay from checkpoints.
    """
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db


@cli.command()
@click.argument("trace_id_a")
@click.argument("trace_id_b")
@click.option("--format", "-f", type=click.Choice(["text", "json", "html", "mermaid"]), default="text")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--semantic/--no-semantic", default=False, help="Include semantic diff")
@click.option("--cost/--no-cost", default=True, help="Include cost comparison")
@click.pass_context
def diff(ctx, trace_id_a, trace_id_b, format, output, semantic, cost):
    """
    Compare two traces and show differences.

    TRACE_ID_A is the baseline trace.
    TRACE_ID_B is the comparison trace.
    """
    async def _diff():
        storage = SQLiteStore(ctx.obj["db_path"])
        await storage.initialize()

        try:
            trace_a = await storage.get_trace(trace_id_a)
            trace_b = await storage.get_trace(trace_id_b)

            if not trace_a:
                click.echo(f"Error: Trace not found: {trace_id_a}", err=True)
                sys.exit(1)
            if not trace_b:
                click.echo(f"Error: Trace not found: {trace_id_b}", err=True)
                sys.exit(1)

            # Structural diff
            engine = StructuralDiffEngine()
            diff_result = engine.diff(trace_a, trace_b)

            # Semantic diff (optional)
            semantic_result = None
            if semantic:
                try:
                    semantic_engine = SemanticDiffEngine()
                    semantic_result = semantic_engine.diff(trace_a, trace_b)
                except ImportError:
                    click.echo("Warning: sentence-transformers not installed, skipping semantic diff", err=True)

            # Cost diff (optional)
            cost_result = None
            if cost:
                cost_engine = CostDiffEngine()
                cost_result = cost_engine.compare(trace_a, trace_b)

            # Render output
            renderer = DiffRenderer(color=format == "text")

            if format == "text":
                result = renderer.render_terminal(diff_result)
                if cost_result:
                    result += "\n" + renderer.render_cost_comparison(cost_result)
            elif format == "json":
                data = {
                    "structural_diff": diff_result.to_dict(),
                }
                if semantic_result:
                    data["semantic_diff"] = semantic_result.to_dict()
                if cost_result:
                    data["cost_diff"] = cost_result.to_dict()
                result = json.dumps(data, indent=2, default=str)
            elif format == "html":
                result = renderer.render_html(diff_result)
            elif format == "mermaid":
                result = renderer.render_mermaid(diff_result)

            if output:
                Path(output).write_text(result)
                click.echo(f"Output written to: {output}")
            else:
                click.echo(result)

        finally:
            await storage.close()

    run_async(_diff())


@cli.command()
@click.option("--name", "-n", help="Filter by trace name")
@click.option("--branch", "-b", default="main", help="Filter by branch")
@click.option("--limit", "-l", default=20, help="Number of traces to show")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def list(ctx, name, branch, limit, format):
    """List recorded traces."""
    async def _list():
        storage = SQLiteStore(ctx.obj["db_path"])
        await storage.initialize()

        try:
            traces = await storage.list_traces(name=name, branch=branch, limit=limit)

            if format == "json":
                data = [t.to_dict() for t in traces]
                click.echo(json.dumps(data, indent=2, default=str))
            else:
                if not traces:
                    click.echo("No traces found.")
                    return

                click.echo(f"{'ID':<40} {'Name':<25} {'Version':<10} {'Status':<10} {'Cost':<10}")
                click.echo("-" * 95)

                for trace in traces:
                    trace_id = trace.trace_id[:36] + "..."
                    name_str = (trace.name[:22] + "...") if len(trace.name) > 25 else trace.name
                    click.echo(
                        f"{trace_id:<40} {name_str:<25} {trace.version:<10} "
                        f"{trace.status:<10} ${trace.total_cost:<.4f}"
                    )

        finally:
            await storage.close()

    run_async(_list())


@cli.command()
@click.argument("trace_id")
@click.option("--format", "-f", type=click.Choice(["text", "json", "tree"]), default="text")
@click.pass_context
def show(ctx, trace_id, format):
    """Show details of a specific trace."""
    async def _show():
        storage = SQLiteStore(ctx.obj["db_path"])
        await storage.initialize()

        try:
            trace = await storage.get_trace(trace_id)

            if not trace:
                click.echo(f"Error: Trace not found: {trace_id}", err=True)
                sys.exit(1)

            if format == "json":
                click.echo(json.dumps(trace.to_dict(), indent=2, default=str))
            elif format == "tree":
                _render_tree(trace)
            else:
                click.echo(f"Trace: {trace.name}")
                click.echo(f"  ID: {trace.trace_id}")
                click.echo(f"  Version: {trace.version}")
                click.echo(f"  Branch: {trace.branch}")
                click.echo(f"  Status: {trace.status}")
                click.echo(f"  Spans: {len(trace.spans)}")
                click.echo(f"  Total Tokens: {trace.total_tokens:,}")
                click.echo(f"  Total Cost: ${trace.total_cost:.4f}")
                click.echo(f"  Duration: {trace.total_duration_ms:.0f}ms")
                click.echo()
                click.echo("Spans:")
                for i, span in enumerate(trace.spans):
                    indent = "  " * (1 if span.parent_span_id else 0)
                    click.echo(f"  {indent}{i+1}. {span.name} ({span.span_type.value})")

        finally:
            await storage.close()

    run_async(_show())


def _render_tree(trace):
    """Render trace as a tree."""
    tree = trace.get_span_tree()

    def print_node(node, prefix="", is_last=True):
        if node["span"]:
            span = node["span"]
            connector = "└── " if is_last else "├── "
            icon = {
                "llm_call": "🤖",
                "tool_call": "🔧",
                "reasoning": "💭",
                "retrieval": "📚",
            }.get(span.span_type.value, "•")

            click.echo(f"{prefix}{connector}{icon} {span.name}")

            child_prefix = prefix + ("    " if is_last else "│   ")
            children = node.get("children", [])
            for i, child in enumerate(children):
                print_node(child, child_prefix, i == len(children) - 1)

    click.echo(f"📋 {trace.name} (v{trace.version})")
    if tree["span"]:
        print_node(tree, is_last=True)
    else:
        for i, child in enumerate(tree.get("children", [])):
            print_node(child, "", i == len(tree["children"]) - 1)


@cli.command()
@click.argument("trace_id")
@click.option("--from-span", "-s", help="Start replay from this span ID")
@click.option("--from-checkpoint", "-c", help="Start replay from this checkpoint ID")
@click.option("--mode", "-m", type=click.Choice(["exact", "live", "hybrid"]), default="exact")
@click.option("--output", "-o", type=click.Path(), help="Output file for replay trace")
@click.pass_context
def replay(ctx, trace_id, from_span, from_checkpoint, mode, output):
    """
    Replay a trace execution.

    Replays the trace from the beginning or from a specific checkpoint.
    """
    async def _replay():
        storage = SQLiteStore(ctx.obj["db_path"])
        await storage.initialize()

        try:
            replay_mode = ReplayMode(mode)
            engine = ReplayEngine(storage, mode=replay_mode)

            if from_span:
                result = await engine.replay_from_span(trace_id, from_span)
            else:
                result = await engine.replay(trace_id, from_checkpoint=from_checkpoint)

            click.echo(f"Replay completed!")
            click.echo(f"  Original Trace: {result.original_trace.trace_id}")
            click.echo(f"  Replay Trace: {result.replay_trace.trace_id}")
            click.echo(f"  Diverged: {result.diverged}")
            if result.divergence_point:
                click.echo(f"  Divergence Point: {result.divergence_point}")
            click.echo(f"  Cost Delta: ${result.cost_delta:+.4f}")

            if output:
                Path(output).write_text(
                    json.dumps(result.to_dict(), indent=2, default=str)
                )
                click.echo(f"  Output: {output}")

        finally:
            await storage.close()

    run_async(_replay())


@cli.command()
@click.argument("trace_id")
@click.pass_context
def checkpoints(ctx, trace_id):
    """List checkpoints for a trace."""
    async def _checkpoints():
        storage = SQLiteStore(ctx.obj["db_path"])
        await storage.initialize()

        try:
            checkpoints = await storage.list_checkpoints(trace_id)

            if not checkpoints:
                click.echo("No checkpoints found.")
                return

            click.echo(f"{'ID':<40} {'Span ID':<40} {'Name':<30}")
            click.echo("-" * 110)

            for cp in checkpoints:
                click.echo(
                    f"{cp['checkpoint_id'][:36]:<40} "
                    f"{cp['span_id'][:36]:<40} "
                    f"{cp['name']:<30}"
                )

        finally:
            await storage.close()

    run_async(_checkpoints())


@cli.command()
@click.argument("trace_id")
@click.option("--model", "-m", help="Target model for cost estimation")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text")
@click.pass_context
def cost(ctx, trace_id, model, format):
    """Analyze costs for a trace."""
    async def _cost():
        storage = SQLiteStore(ctx.obj["db_path"])
        await storage.initialize()

        try:
            trace = await storage.get_trace(trace_id)

            if not trace:
                click.echo(f"Error: Trace not found: {trace_id}", err=True)
                sys.exit(1)

            engine = CostDiffEngine()
            breakdown = engine.analyze_trace(trace)

            if model:
                # Estimate savings with different model
                savings = engine.estimate_cost_savings(trace, model)

                if format == "json":
                    click.echo(json.dumps(savings, indent=2))
                else:
                    click.echo(f"Cost Savings Estimate ({model}):")
                    click.echo(f"  Current Cost: ${savings['current_cost']:.4f}")
                    click.echo(f"  Estimated Cost: ${savings['estimated_cost']:.4f}")
                    click.echo(f"  Potential Savings: ${savings['savings']:.4f} ({savings['savings_percentage']:.1f}%)")
            else:
                if format == "json":
                    click.echo(json.dumps(breakdown.to_dict(), indent=2))
                else:
                    click.echo(f"Cost Breakdown for: {trace.name}")
                    click.echo(f"  Total Cost: ${breakdown.total_cost:.4f}")
                    click.echo(f"  Total Tokens: {breakdown.total_tokens:,}")
                    click.echo()
                    click.echo("By Model:")
                    for model_name, cost in breakdown.cost_by_model.items():
                        click.echo(f"  {model_name}: ${cost:.4f}")
                    click.echo()
                    click.echo("By Type:")
                    for type_name, cost in breakdown.cost_by_type.items():
                        click.echo(f"  {type_name}: ${cost:.4f}")
                    click.echo()
                    click.echo("Top Spans by Cost:")
                    top_spans = engine.get_most_expensive_spans(trace, top_k=5)
                    for span in top_spans:
                        click.echo(f"  {span.span_name}: ${span.total_cost:.4f} ({span.cost_percentage:.1f}%)")

        finally:
            await storage.close()

    run_async(_cost())


@cli.command()
@click.argument("trace_id")
@click.pass_context
def delete(ctx, trace_id):
    """Delete a trace."""
    async def _delete():
        storage = SQLiteStore(ctx.obj["db_path"])
        await storage.initialize()

        try:
            if click.confirm(f"Are you sure you want to delete trace {trace_id}?"):
                deleted = await storage.delete_trace(trace_id)
                if deleted:
                    click.echo("Trace deleted.")
                else:
                    click.echo("Trace not found.")

        finally:
            await storage.close()

    run_async(_delete())


@cli.command()
@click.pass_context
def stats(ctx):
    """Show database statistics."""
    async def _stats():
        storage = SQLiteStore(ctx.obj["db_path"])
        await storage.initialize()

        try:
            stats = await storage.get_statistics()

            click.echo("AgentDiff Database Statistics")
            click.echo("-" * 40)
            click.echo(f"Database: {stats.get('db_path', 'N/A')}")
            click.echo(f"Size: {stats.get('db_size_bytes', 0) / 1024:.1f} KB")
            click.echo(f"Total Traces: {stats.get('trace_count', 0):,}")
            click.echo(f"Total Spans: {stats.get('span_count', 0):,}")
            click.echo(f"Total Tokens: {stats.get('total_tokens', 0):,}")
            click.echo(f"Total Cost: ${stats.get('total_cost', 0):.4f}")

        finally:
            await storage.close()

    run_async(_stats())


@cli.command()
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=8000, help="Port to bind to")
@click.pass_context
def serve(ctx, host, port):
    """Start the AgentDiff web server."""
    try:
        import uvicorn
        from ..api.server import create_app

        app = create_app(ctx.obj["db_path"])
        click.echo(f"Starting AgentDiff server at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        click.echo("Error: uvicorn and fastapi are required for the web server.")
        click.echo("Install with: pip install uvicorn fastapi")
        sys.exit(1)


def main():
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
