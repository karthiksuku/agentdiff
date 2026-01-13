"""
FastAPI server for AgentDiff web interface.
"""

from typing import Optional, List
from contextlib import asynccontextmanager

from ..storage.sqlite_store import SQLiteStore
from ..diff.structural_diff import StructuralDiffEngine
from ..diff.cost_diff import CostDiffEngine


def create_app(db_path: str = "agentdiff.db"):
    """
    Create FastAPI application.

    Args:
        db_path: Path to the database

    Returns:
        FastAPI application
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "FastAPI is required for the web server. "
            "Install with: pip install fastapi uvicorn"
        )

    storage = SQLiteStore(db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize storage on startup."""
        await storage.initialize()
        yield
        await storage.close()

    app = FastAPI(
        title="AgentDiff API",
        description="Visual Diff & Replay Debugger for AI Agents",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Models
    class TraceListResponse(BaseModel):
        traces: List[dict]
        total: int

    class DiffRequest(BaseModel):
        trace_id_a: str
        trace_id_b: str
        include_semantic: bool = False

    class ReplayRequest(BaseModel):
        trace_id: str
        from_span: Optional[str] = None
        mode: str = "exact"

    # Routes
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "AgentDiff API",
            "version": "0.1.0",
            "docs": "/docs",
        }

    @app.get("/traces", response_model=TraceListResponse)
    async def list_traces(
        name: Optional[str] = None,
        branch: str = "main",
        limit: int = 50,
        offset: int = 0,
    ):
        """List all traces."""
        traces = await storage.list_traces(
            name=name,
            branch=branch,
            limit=limit,
            offset=offset,
        )
        return {
            "traces": [t.to_dict() for t in traces],
            "total": len(traces),
        }

    @app.get("/traces/{trace_id}")
    async def get_trace(trace_id: str):
        """Get a specific trace."""
        trace = await storage.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")
        return trace.to_dict()

    @app.delete("/traces/{trace_id}")
    async def delete_trace(trace_id: str):
        """Delete a trace."""
        deleted = await storage.delete_trace(trace_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Trace not found")
        return {"status": "deleted"}

    @app.get("/traces/{trace_id}/spans")
    async def get_trace_spans(trace_id: str):
        """Get all spans for a trace."""
        spans = await storage.get_spans_by_trace(trace_id)
        return {"spans": [s.to_dict() for s in spans]}

    @app.get("/traces/{trace_id}/checkpoints")
    async def get_trace_checkpoints(trace_id: str):
        """Get all checkpoints for a trace."""
        checkpoints = await storage.list_checkpoints(trace_id)
        return {"checkpoints": checkpoints}

    @app.post("/diff")
    async def compute_diff(request: DiffRequest):
        """Compute diff between two traces."""
        trace_a = await storage.get_trace(request.trace_id_a)
        trace_b = await storage.get_trace(request.trace_id_b)

        if not trace_a:
            raise HTTPException(status_code=404, detail=f"Trace not found: {request.trace_id_a}")
        if not trace_b:
            raise HTTPException(status_code=404, detail=f"Trace not found: {request.trace_id_b}")

        engine = StructuralDiffEngine()
        diff = engine.diff(trace_a, trace_b)

        return diff.to_dict()

    @app.get("/traces/{trace_id}/cost")
    async def analyze_cost(trace_id: str):
        """Analyze costs for a trace."""
        trace = await storage.get_trace(trace_id)
        if not trace:
            raise HTTPException(status_code=404, detail="Trace not found")

        engine = CostDiffEngine()
        breakdown = engine.analyze_trace(trace)

        return breakdown.to_dict()

    @app.post("/traces/{trace_id_a}/compare/{trace_id_b}/cost")
    async def compare_costs(trace_id_a: str, trace_id_b: str):
        """Compare costs between two traces."""
        trace_a = await storage.get_trace(trace_id_a)
        trace_b = await storage.get_trace(trace_id_b)

        if not trace_a:
            raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id_a}")
        if not trace_b:
            raise HTTPException(status_code=404, detail=f"Trace not found: {trace_id_b}")

        engine = CostDiffEngine()
        comparison = engine.compare(trace_a, trace_b)

        return comparison.to_dict()

    @app.get("/stats")
    async def get_stats():
        """Get database statistics."""
        return await storage.get_statistics()

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app
