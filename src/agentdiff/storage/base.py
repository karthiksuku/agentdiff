"""
Abstract base class for storage backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from ..core.trace import Trace
from ..core.span import Span


class BaseStorage(ABC):
    """
    Abstract base class for AgentDiff storage backends.

    All storage implementations must implement these methods to provide
    persistence for traces and spans.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the storage backend.

        This should create any necessary tables, indexes, or connections.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        Close the storage backend.

        This should clean up any connections or resources.
        """
        pass

    @abstractmethod
    async def save_trace(self, trace: Trace) -> str:
        """
        Save a complete trace with all its spans.

        Args:
            trace: The trace to save

        Returns:
            The trace ID
        """
        pass

    @abstractmethod
    async def get_trace(self, trace_id: str) -> Optional[Trace]:
        """
        Retrieve a trace by ID.

        Args:
            trace_id: The trace ID to retrieve

        Returns:
            The trace if found, None otherwise
        """
        pass

    @abstractmethod
    async def list_traces(
        self,
        name: Optional[str] = None,
        branch: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Trace]:
        """
        List traces with optional filtering.

        Args:
            name: Filter by trace name
            branch: Filter by branch
            tags: Filter by tags (all must match)
            limit: Maximum number of traces to return
            offset: Number of traces to skip

        Returns:
            List of matching traces
        """
        pass

    @abstractmethod
    async def delete_trace(self, trace_id: str) -> bool:
        """
        Delete a trace and all its spans.

        Args:
            trace_id: The trace ID to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def get_traces_by_name(
        self,
        name: str,
        branch: str = "main",
        limit: int = 10,
    ) -> List[Trace]:
        """
        Get traces by name, useful for comparing versions.

        Args:
            name: The trace name
            branch: The branch to filter by
            limit: Maximum number of traces to return

        Returns:
            List of matching traces, ordered by creation time (newest first)
        """
        pass

    @abstractmethod
    async def save_span(self, span: Span) -> str:
        """
        Save or update a single span.

        Args:
            span: The span to save

        Returns:
            The span ID
        """
        pass

    @abstractmethod
    async def get_span(self, span_id: str) -> Optional[Span]:
        """
        Retrieve a span by ID.

        Args:
            span_id: The span ID to retrieve

        Returns:
            The span if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_spans_by_trace(self, trace_id: str) -> List[Span]:
        """
        Get all spans for a trace.

        Args:
            trace_id: The trace ID

        Returns:
            List of spans for the trace
        """
        pass

    # Optional methods with default implementations

    async def create_checkpoint(
        self,
        trace_id: str,
        span_id: str,
        name: str,
        state_snapshot: Dict[str, Any],
    ) -> str:
        """
        Create a checkpoint for replay.

        Args:
            trace_id: The trace ID
            span_id: The span ID to checkpoint at
            name: Name for the checkpoint
            state_snapshot: State data to save

        Returns:
            The checkpoint ID
        """
        raise NotImplementedError("Checkpoints not supported by this storage backend")

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a checkpoint by ID.

        Args:
            checkpoint_id: The checkpoint ID

        Returns:
            Checkpoint data if found, None otherwise
        """
        raise NotImplementedError("Checkpoints not supported by this storage backend")

    async def list_checkpoints(self, trace_id: str) -> List[Dict[str, Any]]:
        """
        List all checkpoints for a trace.

        Args:
            trace_id: The trace ID

        Returns:
            List of checkpoint data
        """
        raise NotImplementedError("Checkpoints not supported by this storage backend")

    async def save_comparison(
        self,
        trace_id_a: str,
        trace_id_b: str,
        diff_result: Dict[str, Any],
    ) -> str:
        """
        Save a comparison result.

        Args:
            trace_id_a: First trace ID
            trace_id_b: Second trace ID
            diff_result: The diff result data

        Returns:
            The comparison ID
        """
        raise NotImplementedError("Comparisons not supported by this storage backend")

    async def semantic_search_spans(
        self,
        query_embedding: List[float],
        trace_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Span]:
        """
        Search spans by semantic similarity.

        Args:
            query_embedding: The query embedding vector
            trace_id: Optional trace ID to limit search
            limit: Maximum number of results

        Returns:
            List of matching spans
        """
        raise NotImplementedError("Semantic search not supported by this storage backend")

    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with statistics (trace count, span count, etc.)
        """
        return {
            "supported": False,
        }
