"""
SQLite storage backend for AgentDiff.

This is the default storage backend, suitable for local development
and single-user scenarios.
"""

import json
import uuid
import aiosqlite
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from .base import BaseStorage
from ..core.trace import Trace
from ..core.span import Span, SpanType, SpanStatus, TokenUsage


class SQLiteStore(BaseStorage):
    """
    SQLite storage backend.

    Features:
    - File-based storage (no server required)
    - JSON columns for flexible data
    - Full-text search on span content
    - Checkpoint support for replay
    """

    def __init__(
        self,
        db_path: str = "agentdiff.db",
        table_prefix: str = "agentdiff_",
    ):
        """
        Initialize SQLite store.

        Args:
            db_path: Path to the SQLite database file
            table_prefix: Prefix for table names
        """
        self.db_path = Path(db_path)
        self.table_prefix = table_prefix
        self._connection: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Initialize database and create tables."""
        self._connection = await aiosqlite.connect(self.db_path)
        self._connection.row_factory = aiosqlite.Row

        # Enable JSON support
        await self._connection.execute("PRAGMA journal_mode=WAL")

        await self._create_tables()

    async def _create_tables(self) -> None:
        """Create database tables."""
        # Traces table
        await self._connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_prefix}traces (
                trace_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version TEXT DEFAULT '1.0.0',
                branch TEXT DEFAULT 'main',
                parent_trace_id TEXT,
                start_time TEXT,
                end_time TEXT,
                total_tokens INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0,
                total_duration_ms REAL DEFAULT 0.0,
                metadata TEXT DEFAULT '{{}}',
                tags TEXT DEFAULT '[]',
                commit_message TEXT,
                status TEXT DEFAULT 'running',
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (parent_trace_id) REFERENCES {self.table_prefix}traces(trace_id)
            )
        """)

        # Spans table
        await self._connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_prefix}spans (
                span_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                parent_span_id TEXT,
                name TEXT,
                span_type TEXT DEFAULT 'custom',
                status TEXT DEFAULT 'running',
                model TEXT,
                provider TEXT,
                start_time TEXT,
                end_time TEXT,
                duration_ms REAL,
                token_usage TEXT,
                input_data TEXT DEFAULT '{{}}',
                output_data TEXT DEFAULT '{{}}',
                metadata TEXT DEFAULT '{{}}',
                tags TEXT DEFAULT '[]',
                input_embedding TEXT,
                output_embedding TEXT,
                confidence_score REAL,
                alternatives TEXT DEFAULT '[]',
                reasoning TEXT,
                error TEXT,
                error_type TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trace_id) REFERENCES {self.table_prefix}traces(trace_id) ON DELETE CASCADE,
                FOREIGN KEY (parent_span_id) REFERENCES {self.table_prefix}spans(span_id)
            )
        """)

        # Checkpoints table
        await self._connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_prefix}checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                trace_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                name TEXT,
                state_snapshot TEXT DEFAULT '{{}}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trace_id) REFERENCES {self.table_prefix}traces(trace_id) ON DELETE CASCADE,
                FOREIGN KEY (span_id) REFERENCES {self.table_prefix}spans(span_id)
            )
        """)

        # Comparisons table
        await self._connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_prefix}comparisons (
                comparison_id TEXT PRIMARY KEY,
                trace_id_a TEXT NOT NULL,
                trace_id_b TEXT NOT NULL,
                diff_result TEXT DEFAULT '{{}}',
                structural_similarity REAL,
                semantic_similarity REAL,
                cost_delta REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (trace_id_a) REFERENCES {self.table_prefix}traces(trace_id),
                FOREIGN KEY (trace_id_b) REFERENCES {self.table_prefix}traces(trace_id)
            )
        """)

        # Create indexes
        await self._connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_spans_trace_id
            ON {self.table_prefix}spans(trace_id)
        """)

        await self._connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_traces_name
            ON {self.table_prefix}traces(name)
        """)

        await self._connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_traces_branch
            ON {self.table_prefix}traces(branch)
        """)

        await self._connection.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_traces_created
            ON {self.table_prefix}traces(created_at DESC)
        """)

        await self._connection.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def save_trace(self, trace: Trace) -> str:
        """Save a complete trace with all spans."""
        # Insert trace
        await self._connection.execute(
            f"""
            INSERT OR REPLACE INTO {self.table_prefix}traces (
                trace_id, name, version, branch, parent_trace_id,
                start_time, end_time, total_tokens, total_cost,
                total_duration_ms, metadata, tags, commit_message,
                status, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trace.trace_id,
                trace.name,
                trace.version,
                trace.branch,
                trace.parent_trace_id,
                trace.start_time.isoformat() if trace.start_time else None,
                trace.end_time.isoformat() if trace.end_time else None,
                trace.total_tokens,
                trace.total_cost,
                trace.total_duration_ms,
                json.dumps(trace.metadata),
                json.dumps(trace.tags),
                trace.commit_message,
                trace.status,
                trace.error,
            ),
        )

        # Insert spans
        for span in trace.spans:
            await self._insert_span(span)

        await self._connection.commit()
        return trace.trace_id

    async def _insert_span(self, span: Span) -> None:
        """Insert a single span."""
        await self._connection.execute(
            f"""
            INSERT OR REPLACE INTO {self.table_prefix}spans (
                span_id, trace_id, parent_span_id, name, span_type, status,
                model, provider, start_time, end_time, duration_ms,
                token_usage, input_data, output_data, metadata, tags,
                input_embedding, output_embedding, confidence_score,
                alternatives, reasoning, error, error_type
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                span.span_id,
                span.trace_id,
                span.parent_span_id,
                span.name,
                span.span_type.value,
                span.status.value,
                span.model,
                span.provider,
                span.start_time.isoformat() if span.start_time else None,
                span.end_time.isoformat() if span.end_time else None,
                span.duration_ms,
                json.dumps(span.token_usage.to_dict()) if span.token_usage else None,
                json.dumps(span.input_data),
                json.dumps(span.output_data),
                json.dumps(span.metadata),
                json.dumps(span.tags),
                json.dumps(span.input_embedding) if span.input_embedding else None,
                json.dumps(span.output_embedding) if span.output_embedding else None,
                span.confidence_score,
                json.dumps(span.alternatives_considered),
                span.reasoning,
                span.error,
                span.error_type,
            ),
        )

    async def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Retrieve a trace by ID."""
        cursor = await self._connection.execute(
            f"SELECT * FROM {self.table_prefix}traces WHERE trace_id = ?",
            (trace_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        trace = self._row_to_trace(row)

        # Get spans
        cursor = await self._connection.execute(
            f"SELECT * FROM {self.table_prefix}spans WHERE trace_id = ? ORDER BY start_time",
            (trace_id,),
        )
        rows = await cursor.fetchall()
        trace.spans = [self._row_to_span(r) for r in rows]

        return trace

    async def list_traces(
        self,
        name: Optional[str] = None,
        branch: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trace]:
        """List traces with optional filtering."""
        query = f"SELECT * FROM {self.table_prefix}traces WHERE 1=1"
        params: List[Any] = []

        if name:
            query += " AND name = ?"
            params.append(name)

        if branch:
            query += " AND branch = ?"
            params.append(branch)

        # Note: Tag filtering with JSON requires additional handling
        # For now, we'll filter in Python after fetching

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = await self._connection.execute(query, params)
        rows = await cursor.fetchall()

        traces = [self._row_to_trace(r) for r in rows]

        # Filter by tags if specified
        if tags:
            traces = [
                t for t in traces
                if all(tag in t.tags for tag in tags)
            ]

        return traces

    async def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace and all its spans."""
        cursor = await self._connection.execute(
            f"DELETE FROM {self.table_prefix}traces WHERE trace_id = ?",
            (trace_id,),
        )
        await self._connection.commit()
        return cursor.rowcount > 0

    async def get_traces_by_name(
        self,
        name: str,
        branch: str = "main",
        limit: int = 10,
    ) -> List[Trace]:
        """Get traces by name for comparing versions."""
        cursor = await self._connection.execute(
            f"""
            SELECT * FROM {self.table_prefix}traces
            WHERE name = ? AND branch = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (name, branch, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_trace(r) for r in rows]

    async def save_span(self, span: Span) -> str:
        """Save or update a single span."""
        await self._insert_span(span)
        await self._connection.commit()
        return span.span_id

    async def get_span(self, span_id: str) -> Optional[Span]:
        """Retrieve a span by ID."""
        cursor = await self._connection.execute(
            f"SELECT * FROM {self.table_prefix}spans WHERE span_id = ?",
            (span_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_span(row) if row else None

    async def get_spans_by_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        cursor = await self._connection.execute(
            f"SELECT * FROM {self.table_prefix}spans WHERE trace_id = ? ORDER BY start_time",
            (trace_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_span(r) for r in rows]

    async def create_checkpoint(
        self,
        trace_id: str,
        span_id: str,
        name: str,
        state_snapshot: Dict[str, Any],
    ) -> str:
        """Create a checkpoint for replay."""
        checkpoint_id = str(uuid.uuid4())

        await self._connection.execute(
            f"""
            INSERT INTO {self.table_prefix}checkpoints (
                checkpoint_id, trace_id, span_id, name, state_snapshot
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                checkpoint_id,
                trace_id,
                span_id,
                name,
                json.dumps(state_snapshot),
            ),
        )
        await self._connection.commit()
        return checkpoint_id

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get a checkpoint by ID."""
        cursor = await self._connection.execute(
            f"SELECT * FROM {self.table_prefix}checkpoints WHERE checkpoint_id = ?",
            (checkpoint_id,),
        )
        row = await cursor.fetchone()

        if not row:
            return None

        return {
            "checkpoint_id": row["checkpoint_id"],
            "trace_id": row["trace_id"],
            "span_id": row["span_id"],
            "name": row["name"],
            "state_snapshot": json.loads(row["state_snapshot"]),
            "created_at": row["created_at"],
        }

    async def list_checkpoints(self, trace_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a trace."""
        cursor = await self._connection.execute(
            f"SELECT * FROM {self.table_prefix}checkpoints WHERE trace_id = ? ORDER BY created_at",
            (trace_id,),
        )
        rows = await cursor.fetchall()

        return [
            {
                "checkpoint_id": row["checkpoint_id"],
                "trace_id": row["trace_id"],
                "span_id": row["span_id"],
                "name": row["name"],
                "state_snapshot": json.loads(row["state_snapshot"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    async def save_comparison(
        self,
        trace_id_a: str,
        trace_id_b: str,
        diff_result: Dict[str, Any],
    ) -> str:
        """Save a comparison result."""
        comparison_id = str(uuid.uuid4())

        await self._connection.execute(
            f"""
            INSERT INTO {self.table_prefix}comparisons (
                comparison_id, trace_id_a, trace_id_b, diff_result,
                structural_similarity, semantic_similarity, cost_delta
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                comparison_id,
                trace_id_a,
                trace_id_b,
                json.dumps(diff_result),
                diff_result.get("structural_similarity"),
                diff_result.get("semantic_similarity"),
                diff_result.get("cost_delta"),
            ),
        )
        await self._connection.commit()
        return comparison_id

    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        cursor = await self._connection.execute(
            f"SELECT COUNT(*) as count FROM {self.table_prefix}traces"
        )
        trace_count = (await cursor.fetchone())["count"]

        cursor = await self._connection.execute(
            f"SELECT COUNT(*) as count FROM {self.table_prefix}spans"
        )
        span_count = (await cursor.fetchone())["count"]

        cursor = await self._connection.execute(
            f"SELECT SUM(total_tokens) as total, SUM(total_cost) as cost FROM {self.table_prefix}traces"
        )
        row = await cursor.fetchone()

        return {
            "supported": True,
            "trace_count": trace_count,
            "span_count": span_count,
            "total_tokens": row["total"] or 0,
            "total_cost": row["cost"] or 0.0,
            "db_path": str(self.db_path),
            "db_size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
        }

    def _row_to_trace(self, row: aiosqlite.Row) -> Trace:
        """Convert database row to Trace object."""
        trace = Trace(
            trace_id=row["trace_id"],
            name=row["name"],
            version=row["version"] or "1.0.0",
            branch=row["branch"] or "main",
            parent_trace_id=row["parent_trace_id"],
            total_tokens=row["total_tokens"] or 0,
            total_cost=row["total_cost"] or 0.0,
            total_duration_ms=row["total_duration_ms"] or 0.0,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            tags=json.loads(row["tags"]) if row["tags"] else [],
            commit_message=row["commit_message"],
            status=row["status"] or "running",
            error=row["error"],
        )

        if row["start_time"]:
            trace.start_time = datetime.fromisoformat(row["start_time"])
        if row["end_time"]:
            trace.end_time = datetime.fromisoformat(row["end_time"])

        return trace

    def _row_to_span(self, row: aiosqlite.Row) -> Span:
        """Convert database row to Span object."""
        token_usage = None
        if row["token_usage"]:
            token_usage = TokenUsage.from_dict(json.loads(row["token_usage"]))

        span = Span(
            span_id=row["span_id"],
            trace_id=row["trace_id"],
            parent_span_id=row["parent_span_id"],
            name=row["name"] or "",
            span_type=SpanType(row["span_type"]) if row["span_type"] else SpanType.CUSTOM,
            status=SpanStatus(row["status"]) if row["status"] else SpanStatus.RUNNING,
            model=row["model"],
            provider=row["provider"],
            duration_ms=row["duration_ms"],
            token_usage=token_usage,
            input_data=json.loads(row["input_data"]) if row["input_data"] else {},
            output_data=json.loads(row["output_data"]) if row["output_data"] else {},
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            tags=json.loads(row["tags"]) if row["tags"] else [],
            input_embedding=json.loads(row["input_embedding"]) if row["input_embedding"] else None,
            output_embedding=json.loads(row["output_embedding"]) if row["output_embedding"] else None,
            confidence_score=row["confidence_score"],
            alternatives_considered=json.loads(row["alternatives"]) if row["alternatives"] else [],
            reasoning=row["reasoning"],
            error=row["error"],
            error_type=row["error_type"],
        )

        if row["start_time"]:
            span.start_time = datetime.fromisoformat(row["start_time"])
        if row["end_time"]:
            span.end_time = datetime.fromisoformat(row["end_time"])

        return span
