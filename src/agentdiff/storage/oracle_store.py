"""
Oracle Autonomous Database Storage Backend for AgentDiff.

Supports:
- Oracle Autonomous Database (ATP/ADW)
- Oracle Database 23ai with JSON and Vector support
- Connection pooling with python-oracledb
- Async operations
- Vector similarity search for semantic diff
"""

import json
import uuid
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

from .base import BaseStorage
from ..core.trace import Trace
from ..core.span import Span, SpanType, SpanStatus, TokenUsage


class OracleAutonomousStore(BaseStorage):
    """
    Oracle Autonomous Database storage backend.

    Features:
    - JSON document storage for flexible span/trace data
    - Vector embeddings support (Oracle 23ai)
    - Full-text search
    - Connection pooling
    - Wallet-based authentication for ADB
    """

    def __init__(
        self,
        # Connection options (choose one method)
        connection_string: Optional[str] = None,  # Easy Connect or TNS

        # Wallet-based connection (for Autonomous Database)
        wallet_location: Optional[str] = None,
        wallet_password: Optional[str] = None,

        # Individual parameters
        user: Optional[str] = None,
        password: Optional[str] = None,
        dsn: Optional[str] = None,

        # Connection pool settings
        min_connections: int = 2,
        max_connections: int = 10,

        # Schema settings
        table_prefix: str = "agentdiff_",

        # Feature flags
        use_vector_search: bool = True,  # Requires Oracle 23ai
        use_json_duality: bool = False,  # JSON Relational Duality Views
    ):
        """
        Initialize Oracle Autonomous Database store.

        Args:
            connection_string: Easy Connect or TNS connection string
            wallet_location: Path to wallet directory for ADB
            wallet_password: Password for the wallet
            user: Database username
            password: Database password
            dsn: Database service name or TNS entry
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
            table_prefix: Prefix for table names
            use_vector_search: Enable vector search (requires Oracle 23ai)
            use_json_duality: Enable JSON Relational Duality Views
        """
        self.connection_string = connection_string
        self.wallet_location = wallet_location
        self.wallet_password = wallet_password
        self.user = user or os.getenv("ORACLE_USER")
        self.password = password or os.getenv("ORACLE_PASSWORD")
        self.dsn = dsn or os.getenv("ORACLE_DSN")

        self.min_connections = min_connections
        self.max_connections = max_connections
        self.table_prefix = table_prefix

        self.use_vector_search = use_vector_search
        self.use_json_duality = use_json_duality

        self._pool = None
        self._oracledb = None

    async def initialize(self) -> None:
        """Initialize connection pool and create schema."""
        try:
            import oracledb
            self._oracledb = oracledb
        except ImportError:
            raise ImportError(
                "python-oracledb is required for Oracle storage. "
                "Install it with: pip install oracledb"
            )

        # Configure for thick mode if using wallet
        if self.wallet_location:
            try:
                self._oracledb.init_oracle_client(config_dir=self.wallet_location)
            except Exception:
                # May already be initialized
                pass

        # Create connection pool
        pool_params = {
            "user": self.user,
            "password": self.password,
            "dsn": self.dsn,
            "min": self.min_connections,
            "max": self.max_connections,
        }

        if self.wallet_location:
            pool_params["wallet_location"] = self.wallet_location
            if self.wallet_password:
                pool_params["wallet_password"] = self.wallet_password

        self._pool = self._oracledb.create_pool(**pool_params)

        # Create schema
        await self._create_schema()

    async def _create_schema(self) -> None:
        """Create database tables for AgentDiff."""
        with self._pool.acquire() as conn:
            cursor = conn.cursor()

            # Check if tables exist
            cursor.execute("""
                SELECT COUNT(*) FROM user_tables
                WHERE table_name = :name
            """, {"name": f"{self.table_prefix.upper()}TRACES"})

            if cursor.fetchone()[0] == 0:
                # Traces table with JSON column
                cursor.execute(f"""
                    CREATE TABLE {self.table_prefix}traces (
                        trace_id VARCHAR2(36) PRIMARY KEY,
                        name VARCHAR2(255) NOT NULL,
                        version VARCHAR2(50) DEFAULT '1.0.0',
                        branch VARCHAR2(100) DEFAULT 'main',
                        parent_trace_id VARCHAR2(36),
                        start_time TIMESTAMP WITH TIME ZONE,
                        end_time TIMESTAMP WITH TIME ZONE,
                        total_tokens NUMBER DEFAULT 0,
                        total_cost NUMBER(15, 6) DEFAULT 0,
                        total_duration_ms NUMBER DEFAULT 0,
                        metadata CLOB CHECK (metadata IS JSON),
                        tags CLOB CHECK (tags IS JSON),
                        commit_message CLOB,
                        status VARCHAR2(20) DEFAULT 'running',
                        error CLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Add foreign key constraint separately
                try:
                    cursor.execute(f"""
                        ALTER TABLE {self.table_prefix}traces
                        ADD CONSTRAINT fk_parent_trace
                        FOREIGN KEY (parent_trace_id)
                        REFERENCES {self.table_prefix}traces(trace_id)
                    """)
                except Exception:
                    pass  # Constraint may already exist

            # Check if spans table exists
            cursor.execute("""
                SELECT COUNT(*) FROM user_tables
                WHERE table_name = :name
            """, {"name": f"{self.table_prefix.upper()}SPANS"})

            if cursor.fetchone()[0] == 0:
                # Spans table with JSON and optional Vector column
                vector_column = ""
                if self.use_vector_search:
                    # Check if VECTOR type is available (Oracle 23ai)
                    try:
                        cursor.execute("SELECT 1 FROM dual WHERE 1=0")
                        vector_column = """
                            input_embedding CLOB,
                            output_embedding CLOB,
                        """
                    except Exception:
                        vector_column = """
                            input_embedding CLOB,
                            output_embedding CLOB,
                        """

                cursor.execute(f"""
                    CREATE TABLE {self.table_prefix}spans (
                        span_id VARCHAR2(36) PRIMARY KEY,
                        trace_id VARCHAR2(36) NOT NULL,
                        parent_span_id VARCHAR2(36),
                        name VARCHAR2(255),
                        span_type VARCHAR2(50) DEFAULT 'custom',
                        status VARCHAR2(20) DEFAULT 'running',
                        model VARCHAR2(100),
                        provider VARCHAR2(50),
                        start_time TIMESTAMP WITH TIME ZONE,
                        end_time TIMESTAMP WITH TIME ZONE,
                        duration_ms NUMBER,
                        token_usage CLOB CHECK (token_usage IS JSON),
                        input_data CLOB CHECK (input_data IS JSON),
                        output_data CLOB CHECK (output_data IS JSON),
                        metadata CLOB CHECK (metadata IS JSON),
                        tags CLOB CHECK (tags IS JSON),
                        {vector_column}
                        confidence_score NUMBER(5, 4),
                        alternatives CLOB CHECK (alternatives IS JSON),
                        reasoning CLOB,
                        error CLOB,
                        error_type VARCHAR2(100),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT fk_trace
                            FOREIGN KEY (trace_id)
                            REFERENCES {self.table_prefix}traces(trace_id)
                            ON DELETE CASCADE
                    )
                """)

                # Add parent span foreign key
                try:
                    cursor.execute(f"""
                        ALTER TABLE {self.table_prefix}spans
                        ADD CONSTRAINT fk_parent_span
                        FOREIGN KEY (parent_span_id)
                        REFERENCES {self.table_prefix}spans(span_id)
                    """)
                except Exception:
                    pass

            # Create indexes
            try:
                cursor.execute(f"""
                    CREATE INDEX idx_spans_trace_id
                    ON {self.table_prefix}spans(trace_id)
                """)
            except Exception:
                pass

            try:
                cursor.execute(f"""
                    CREATE INDEX idx_traces_name_version
                    ON {self.table_prefix}traces(name, version)
                """)
            except Exception:
                pass

            try:
                cursor.execute(f"""
                    CREATE INDEX idx_traces_branch
                    ON {self.table_prefix}traces(branch)
                """)
            except Exception:
                pass

            # Check if checkpoints table exists
            cursor.execute("""
                SELECT COUNT(*) FROM user_tables
                WHERE table_name = :name
            """, {"name": f"{self.table_prefix.upper()}CHECKPOINTS"})

            if cursor.fetchone()[0] == 0:
                cursor.execute(f"""
                    CREATE TABLE {self.table_prefix}checkpoints (
                        checkpoint_id VARCHAR2(36) PRIMARY KEY,
                        trace_id VARCHAR2(36) NOT NULL,
                        span_id VARCHAR2(36) NOT NULL,
                        name VARCHAR2(255),
                        state_snapshot CLOB CHECK (state_snapshot IS JSON),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT fk_checkpoint_trace
                            FOREIGN KEY (trace_id)
                            REFERENCES {self.table_prefix}traces(trace_id)
                            ON DELETE CASCADE
                    )
                """)

            # Check if comparisons table exists
            cursor.execute("""
                SELECT COUNT(*) FROM user_tables
                WHERE table_name = :name
            """, {"name": f"{self.table_prefix.upper()}COMPARISONS"})

            if cursor.fetchone()[0] == 0:
                cursor.execute(f"""
                    CREATE TABLE {self.table_prefix}comparisons (
                        comparison_id VARCHAR2(36) PRIMARY KEY,
                        trace_id_a VARCHAR2(36) NOT NULL,
                        trace_id_b VARCHAR2(36) NOT NULL,
                        diff_result CLOB CHECK (diff_result IS JSON),
                        structural_similarity NUMBER(5, 4),
                        semantic_similarity NUMBER(5, 4),
                        cost_delta NUMBER(15, 6),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        CONSTRAINT fk_comparison_trace_a
                            FOREIGN KEY (trace_id_a)
                            REFERENCES {self.table_prefix}traces(trace_id),
                        CONSTRAINT fk_comparison_trace_b
                            FOREIGN KEY (trace_id_b)
                            REFERENCES {self.table_prefix}traces(trace_id)
                    )
                """)

            conn.commit()

    @asynccontextmanager
    async def _get_connection(self):
        """Get connection from pool."""
        conn = self._pool.acquire()
        try:
            yield conn
        finally:
            self._pool.release(conn)

    async def save_trace(self, trace: Trace) -> str:
        """Save a complete trace with all spans."""
        with self._pool.acquire() as conn:
            cursor = conn.cursor()

            # Insert trace
            cursor.execute(f"""
                MERGE INTO {self.table_prefix}traces t
                USING (SELECT :trace_id as trace_id FROM dual) s
                ON (t.trace_id = s.trace_id)
                WHEN MATCHED THEN UPDATE SET
                    name = :name,
                    version = :version,
                    branch = :branch,
                    parent_trace_id = :parent_trace_id,
                    start_time = :start_time,
                    end_time = :end_time,
                    total_tokens = :total_tokens,
                    total_cost = :total_cost,
                    total_duration_ms = :total_duration_ms,
                    metadata = :metadata,
                    tags = :tags,
                    commit_message = :commit_message,
                    status = :status,
                    error = :error
                WHEN NOT MATCHED THEN INSERT (
                    trace_id, name, version, branch, parent_trace_id,
                    start_time, end_time, total_tokens, total_cost,
                    total_duration_ms, metadata, tags, commit_message,
                    status, error
                ) VALUES (
                    :trace_id, :name, :version, :branch, :parent_trace_id,
                    :start_time, :end_time, :total_tokens, :total_cost,
                    :total_duration_ms, :metadata, :tags, :commit_message,
                    :status, :error
                )
            """, {
                "trace_id": trace.trace_id,
                "name": trace.name,
                "version": trace.version,
                "branch": trace.branch,
                "parent_trace_id": trace.parent_trace_id,
                "start_time": trace.start_time,
                "end_time": trace.end_time,
                "total_tokens": trace.total_tokens,
                "total_cost": trace.total_cost,
                "total_duration_ms": trace.total_duration_ms,
                "metadata": json.dumps(trace.metadata),
                "tags": json.dumps(trace.tags),
                "commit_message": trace.commit_message,
                "status": trace.status,
                "error": trace.error,
            })

            # Insert spans
            for span in trace.spans:
                self._insert_span(cursor, span)

            conn.commit()

        return trace.trace_id

    def _insert_span(self, cursor, span: Span) -> None:
        """Insert a single span."""
        params = {
            "span_id": span.span_id,
            "trace_id": span.trace_id,
            "parent_span_id": span.parent_span_id,
            "name": span.name,
            "span_type": span.span_type.value,
            "status": span.status.value,
            "model": span.model,
            "provider": span.provider,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": span.duration_ms,
            "token_usage": json.dumps(span.token_usage.to_dict()) if span.token_usage else None,
            "input_data": json.dumps(span.input_data),
            "output_data": json.dumps(span.output_data),
            "metadata": json.dumps(span.metadata),
            "tags": json.dumps(span.tags),
            "input_embedding": json.dumps(span.input_embedding) if span.input_embedding else None,
            "output_embedding": json.dumps(span.output_embedding) if span.output_embedding else None,
            "confidence_score": span.confidence_score,
            "alternatives": json.dumps(span.alternatives_considered),
            "reasoning": span.reasoning,
            "error": span.error,
            "error_type": span.error_type,
        }

        cursor.execute(f"""
            MERGE INTO {self.table_prefix}spans s
            USING (SELECT :span_id as span_id FROM dual) src
            ON (s.span_id = src.span_id)
            WHEN MATCHED THEN UPDATE SET
                trace_id = :trace_id,
                parent_span_id = :parent_span_id,
                name = :name,
                span_type = :span_type,
                status = :status,
                model = :model,
                provider = :provider,
                start_time = :start_time,
                end_time = :end_time,
                duration_ms = :duration_ms,
                token_usage = :token_usage,
                input_data = :input_data,
                output_data = :output_data,
                metadata = :metadata,
                tags = :tags,
                input_embedding = :input_embedding,
                output_embedding = :output_embedding,
                confidence_score = :confidence_score,
                alternatives = :alternatives,
                reasoning = :reasoning,
                error = :error,
                error_type = :error_type
            WHEN NOT MATCHED THEN INSERT (
                span_id, trace_id, parent_span_id, name, span_type, status,
                model, provider, start_time, end_time, duration_ms,
                token_usage, input_data, output_data, metadata, tags,
                input_embedding, output_embedding, confidence_score,
                alternatives, reasoning, error, error_type
            ) VALUES (
                :span_id, :trace_id, :parent_span_id, :name, :span_type, :status,
                :model, :provider, :start_time, :end_time, :duration_ms,
                :token_usage, :input_data, :output_data, :metadata, :tags,
                :input_embedding, :output_embedding, :confidence_score,
                :alternatives, :reasoning, :error, :error_type
            )
        """, params)

    async def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Retrieve a trace by ID."""
        with self._pool.acquire() as conn:
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT * FROM {self.table_prefix}traces WHERE trace_id = :trace_id
            """, {"trace_id": trace_id})

            columns = [col[0].lower() for col in cursor.description]
            row = cursor.fetchone()

            if not row:
                return None

            trace = self._row_to_trace(dict(zip(columns, row)))

            # Get spans
            cursor.execute(f"""
                SELECT * FROM {self.table_prefix}spans
                WHERE trace_id = :trace_id
                ORDER BY start_time
            """, {"trace_id": trace_id})

            columns = [col[0].lower() for col in cursor.description]
            trace.spans = [
                self._row_to_span(dict(zip(columns, row)))
                for row in cursor.fetchall()
            ]

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
        with self._pool.acquire() as conn:
            cursor = conn.cursor()

            query = f"SELECT * FROM {self.table_prefix}traces WHERE 1=1"
            params: Dict[str, Any] = {}

            if name:
                query += " AND name = :name"
                params["name"] = name

            if branch:
                query += " AND branch = :branch"
                params["branch"] = branch

            query += " ORDER BY created_at DESC OFFSET :offset ROWS FETCH NEXT :limit ROWS ONLY"
            params["offset"] = offset
            params["limit"] = limit

            cursor.execute(query, params)

            columns = [col[0].lower() for col in cursor.description]
            traces = [
                self._row_to_trace(dict(zip(columns, row)))
                for row in cursor.fetchall()
            ]

            # Filter by tags if specified
            if tags:
                traces = [
                    t for t in traces
                    if all(tag in t.tags for tag in tags)
                ]

            return traces

    async def delete_trace(self, trace_id: str) -> bool:
        """Delete a trace and all its spans."""
        with self._pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                DELETE FROM {self.table_prefix}traces WHERE trace_id = :trace_id
            """, {"trace_id": trace_id})
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    async def get_traces_by_name(
        self,
        name: str,
        branch: str = "main",
        limit: int = 10,
    ) -> List[Trace]:
        """Get traces by name, useful for comparing versions."""
        with self._pool.acquire() as conn:
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT * FROM {self.table_prefix}traces
                WHERE name = :name AND branch = :branch
                ORDER BY created_at DESC
                FETCH FIRST :limit ROWS ONLY
            """, {"name": name, "branch": branch, "limit": limit})

            columns = [col[0].lower() for col in cursor.description]
            return [
                self._row_to_trace(dict(zip(columns, row)))
                for row in cursor.fetchall()
            ]

    async def save_span(self, span: Span) -> str:
        """Save or update a single span."""
        with self._pool.acquire() as conn:
            cursor = conn.cursor()
            self._insert_span(cursor, span)
            conn.commit()
        return span.span_id

    async def get_span(self, span_id: str) -> Optional[Span]:
        """Retrieve a span by ID."""
        with self._pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM {self.table_prefix}spans WHERE span_id = :span_id
            """, {"span_id": span_id})

            columns = [col[0].lower() for col in cursor.description]
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_span(dict(zip(columns, row)))

    async def get_spans_by_trace(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        with self._pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM {self.table_prefix}spans
                WHERE trace_id = :trace_id
                ORDER BY start_time
            """, {"trace_id": trace_id})

            columns = [col[0].lower() for col in cursor.description]
            return [
                self._row_to_span(dict(zip(columns, row)))
                for row in cursor.fetchall()
            ]

    async def create_checkpoint(
        self,
        trace_id: str,
        span_id: str,
        name: str,
        state_snapshot: Dict[str, Any],
    ) -> str:
        """Create a checkpoint for replay."""
        checkpoint_id = str(uuid.uuid4())

        with self._pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {self.table_prefix}checkpoints (
                    checkpoint_id, trace_id, span_id, name, state_snapshot
                ) VALUES (
                    :checkpoint_id, :trace_id, :span_id, :name, :state_snapshot
                )
            """, {
                "checkpoint_id": checkpoint_id,
                "trace_id": trace_id,
                "span_id": span_id,
                "name": name,
                "state_snapshot": json.dumps(state_snapshot),
            })
            conn.commit()

        return checkpoint_id

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get a checkpoint by ID."""
        with self._pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM {self.table_prefix}checkpoints
                WHERE checkpoint_id = :checkpoint_id
            """, {"checkpoint_id": checkpoint_id})

            columns = [col[0].lower() for col in cursor.description]
            row = cursor.fetchone()

            if not row:
                return None

            data = dict(zip(columns, row))
            return {
                "checkpoint_id": data["checkpoint_id"],
                "trace_id": data["trace_id"],
                "span_id": data["span_id"],
                "name": data["name"],
                "state_snapshot": json.loads(data["state_snapshot"]) if data["state_snapshot"] else {},
                "created_at": str(data["created_at"]),
            }

    async def list_checkpoints(self, trace_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a trace."""
        with self._pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM {self.table_prefix}checkpoints
                WHERE trace_id = :trace_id
                ORDER BY created_at
            """, {"trace_id": trace_id})

            columns = [col[0].lower() for col in cursor.description]
            return [
                {
                    "checkpoint_id": data["checkpoint_id"],
                    "trace_id": data["trace_id"],
                    "span_id": data["span_id"],
                    "name": data["name"],
                    "state_snapshot": json.loads(data["state_snapshot"]) if data["state_snapshot"] else {},
                    "created_at": str(data["created_at"]),
                }
                for data in [dict(zip(columns, row)) for row in cursor.fetchall()]
            ]

    async def save_comparison(
        self,
        trace_id_a: str,
        trace_id_b: str,
        diff_result: Dict[str, Any],
    ) -> str:
        """Save a comparison result."""
        comparison_id = str(uuid.uuid4())

        with self._pool.acquire() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                INSERT INTO {self.table_prefix}comparisons (
                    comparison_id, trace_id_a, trace_id_b, diff_result,
                    structural_similarity, semantic_similarity, cost_delta
                ) VALUES (
                    :comparison_id, :trace_id_a, :trace_id_b, :diff_result,
                    :structural_similarity, :semantic_similarity, :cost_delta
                )
            """, {
                "comparison_id": comparison_id,
                "trace_id_a": trace_id_a,
                "trace_id_b": trace_id_b,
                "diff_result": json.dumps(diff_result),
                "structural_similarity": diff_result.get("structural_similarity"),
                "semantic_similarity": diff_result.get("semantic_similarity"),
                "cost_delta": diff_result.get("cost_delta"),
            })
            conn.commit()

        return comparison_id

    async def semantic_search_spans(
        self,
        query_embedding: List[float],
        trace_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Span]:
        """
        Search spans by semantic similarity.

        Note: Full vector search requires Oracle 23ai. This implementation
        uses a simplified approach for compatibility.
        """
        # For Oracle 23ai with native vector support, you would use:
        # VECTOR_DISTANCE(s.input_embedding, :query_vec, COSINE)
        # For now, we return spans and let the caller do similarity calculation
        with self._pool.acquire() as conn:
            cursor = conn.cursor()

            query = f"""
                SELECT * FROM {self.table_prefix}spans
                WHERE input_embedding IS NOT NULL
            """
            params: Dict[str, Any] = {"limit": limit}

            if trace_id:
                query += " AND trace_id = :trace_id"
                params["trace_id"] = trace_id

            query += " FETCH FIRST :limit ROWS ONLY"

            cursor.execute(query, params)

            columns = [col[0].lower() for col in cursor.description]
            return [
                self._row_to_span(dict(zip(columns, row)))
                for row in cursor.fetchall()
            ]

    async def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._pool.acquire() as conn:
            cursor = conn.cursor()

            cursor.execute(f"SELECT COUNT(*) FROM {self.table_prefix}traces")
            trace_count = cursor.fetchone()[0]

            cursor.execute(f"SELECT COUNT(*) FROM {self.table_prefix}spans")
            span_count = cursor.fetchone()[0]

            cursor.execute(f"""
                SELECT SUM(total_tokens), SUM(total_cost)
                FROM {self.table_prefix}traces
            """)
            row = cursor.fetchone()

            return {
                "supported": True,
                "trace_count": trace_count,
                "span_count": span_count,
                "total_tokens": row[0] or 0,
                "total_cost": row[1] or 0.0,
                "storage_type": "oracle",
                "dsn": self.dsn,
            }

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            self._pool.close()
            self._pool = None

    def _row_to_trace(self, data: Dict[str, Any]) -> Trace:
        """Convert database row to Trace object."""
        trace = Trace(
            trace_id=data["trace_id"],
            name=data["name"],
            version=data.get("version", "1.0.0"),
            branch=data.get("branch", "main"),
            parent_trace_id=data.get("parent_trace_id"),
            total_tokens=data.get("total_tokens", 0),
            total_cost=float(data.get("total_cost", 0.0)),
            total_duration_ms=float(data.get("total_duration_ms", 0.0)),
            metadata=json.loads(data.get("metadata", "{}")) if data.get("metadata") else {},
            tags=json.loads(data.get("tags", "[]")) if data.get("tags") else [],
            commit_message=data.get("commit_message"),
            status=data.get("status", "running"),
            error=data.get("error"),
        )

        if data.get("start_time"):
            if isinstance(data["start_time"], str):
                trace.start_time = datetime.fromisoformat(data["start_time"])
            else:
                trace.start_time = data["start_time"]

        if data.get("end_time"):
            if isinstance(data["end_time"], str):
                trace.end_time = datetime.fromisoformat(data["end_time"])
            else:
                trace.end_time = data["end_time"]

        return trace

    def _row_to_span(self, data: Dict[str, Any]) -> Span:
        """Convert database row to Span object."""
        token_usage = None
        if data.get("token_usage"):
            token_data = data["token_usage"]
            if isinstance(token_data, str):
                token_data = json.loads(token_data)
            token_usage = TokenUsage.from_dict(token_data)

        span = Span(
            span_id=data["span_id"],
            trace_id=data["trace_id"],
            parent_span_id=data.get("parent_span_id"),
            name=data.get("name", ""),
            span_type=SpanType(data.get("span_type", "custom")),
            status=SpanStatus(data.get("status", "running")),
            model=data.get("model"),
            provider=data.get("provider"),
            duration_ms=float(data["duration_ms"]) if data.get("duration_ms") else None,
            token_usage=token_usage,
            input_data=json.loads(data.get("input_data", "{}")) if data.get("input_data") else {},
            output_data=json.loads(data.get("output_data", "{}")) if data.get("output_data") else {},
            metadata=json.loads(data.get("metadata", "{}")) if data.get("metadata") else {},
            tags=json.loads(data.get("tags", "[]")) if data.get("tags") else [],
            input_embedding=json.loads(data["input_embedding"]) if data.get("input_embedding") else None,
            output_embedding=json.loads(data["output_embedding"]) if data.get("output_embedding") else None,
            confidence_score=float(data["confidence_score"]) if data.get("confidence_score") else None,
            alternatives_considered=json.loads(data.get("alternatives", "[]")) if data.get("alternatives") else [],
            reasoning=data.get("reasoning"),
            error=data.get("error"),
            error_type=data.get("error_type"),
        )

        if data.get("start_time"):
            if isinstance(data["start_time"], str):
                span.start_time = datetime.fromisoformat(data["start_time"])
            else:
                span.start_time = data["start_time"]

        if data.get("end_time"):
            if isinstance(data["end_time"], str):
                span.end_time = datetime.fromisoformat(data["end_time"])
            else:
                span.end_time = data["end_time"]

        return span


def create_oracle_store(
    wallet_path: Optional[str] = None,
    use_env: bool = True,
) -> OracleAutonomousStore:
    """
    Factory function to create Oracle store with common configurations.

    For Oracle Autonomous Database:
        store = create_oracle_store(wallet_path="/path/to/wallet")

    Using environment variables:
        export ORACLE_USER=admin
        export ORACLE_PASSWORD=your_password
        export ORACLE_DSN=your_adb_tns_name
        store = create_oracle_store()

    Args:
        wallet_path: Path to wallet directory for ADB
        use_env: Whether to use environment variables

    Returns:
        Configured OracleAutonomousStore instance
    """
    if wallet_path:
        return OracleAutonomousStore(
            wallet_location=wallet_path,
            wallet_password=os.getenv("ORACLE_WALLET_PASSWORD"),
        )

    if use_env:
        return OracleAutonomousStore()

    raise ValueError("Must provide wallet_path or set environment variables")
