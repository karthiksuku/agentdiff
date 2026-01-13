"""
Checkpoint management for replay functionality.
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import uuid

from ..core.trace import Trace
from ..core.span import Span
from ..storage.base import BaseStorage


@dataclass
class Checkpoint:
    """Represents a checkpoint in a trace execution."""
    checkpoint_id: str
    trace_id: str
    span_id: str
    name: str
    state_snapshot: Dict[str, Any]
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "name": self.name,
            "state_snapshot": self.state_snapshot,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            name=data.get("name", ""),
            state_snapshot=data.get("state_snapshot", {}),
            created_at=data.get("created_at"),
        )


class CheckpointManager:
    """
    Manages checkpoints for trace replay.

    Checkpoints allow saving and restoring agent state at specific
    points in the execution, enabling replay from any decision point.
    """

    def __init__(self, storage: BaseStorage):
        """
        Initialize the checkpoint manager.

        Args:
            storage: Storage backend
        """
        self.storage = storage

    async def create_checkpoint(
        self,
        trace: Trace,
        span: Span,
        name: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """
        Create a checkpoint at a specific span.

        Args:
            trace: The trace
            span: The span to checkpoint
            name: Optional name for the checkpoint
            state: Optional state snapshot to save

        Returns:
            Created Checkpoint
        """
        checkpoint_name = name or f"checkpoint_{span.name[:20]}_{span.span_id[:8]}"

        # Build state snapshot
        state_snapshot = state or {}

        # Include span context
        state_snapshot["span_context"] = {
            "span_id": span.span_id,
            "span_name": span.name,
            "span_type": span.span_type.value,
            "input_data": span.input_data,
            "output_data": span.output_data,
        }

        # Include parent chain
        parent_chain = []
        current = span
        while current.parent_span_id:
            parent = trace.get_span(current.parent_span_id)
            if parent:
                parent_chain.append({
                    "span_id": parent.span_id,
                    "name": parent.name,
                })
                current = parent
            else:
                break
        state_snapshot["parent_chain"] = parent_chain

        # Save checkpoint
        checkpoint_id = await self.storage.create_checkpoint(
            trace_id=trace.trace_id,
            span_id=span.span_id,
            name=checkpoint_name,
            state_snapshot=state_snapshot,
        )

        return Checkpoint(
            checkpoint_id=checkpoint_id,
            trace_id=trace.trace_id,
            span_id=span.span_id,
            name=checkpoint_name,
            state_snapshot=state_snapshot,
        )

    async def get_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """
        Get a checkpoint by ID.

        Args:
            checkpoint_id: The checkpoint ID

        Returns:
            Checkpoint if found, None otherwise
        """
        data = await self.storage.get_checkpoint(checkpoint_id)
        if data:
            return Checkpoint.from_dict(data)
        return None

    async def list_checkpoints(self, trace_id: str) -> List[Checkpoint]:
        """
        List all checkpoints for a trace.

        Args:
            trace_id: The trace ID

        Returns:
            List of checkpoints
        """
        data_list = await self.storage.list_checkpoints(trace_id)
        return [Checkpoint.from_dict(data) for data in data_list]

    async def auto_checkpoint(
        self,
        trace: Trace,
        checkpoint_types: Optional[List[str]] = None,
    ) -> List[Checkpoint]:
        """
        Automatically create checkpoints at key decision points.

        Args:
            trace: The trace to checkpoint
            checkpoint_types: Types of spans to checkpoint (default: tool_call, reasoning)

        Returns:
            List of created checkpoints
        """
        if checkpoint_types is None:
            checkpoint_types = ["tool_call", "reasoning", "llm_call"]

        checkpoints = []

        for span in trace.spans:
            if span.span_type.value in checkpoint_types:
                checkpoint = await self.create_checkpoint(
                    trace=trace,
                    span=span,
                    name=f"auto_{span.span_type.value}_{span.span_id[:8]}",
                )
                checkpoints.append(checkpoint)

        return checkpoints

    async def find_nearest_checkpoint(
        self,
        trace_id: str,
        target_span_id: str,
    ) -> Optional[Checkpoint]:
        """
        Find the nearest checkpoint before a target span.

        Args:
            trace_id: The trace ID
            target_span_id: The target span ID

        Returns:
            Nearest checkpoint if found
        """
        trace = await self.storage.get_trace(trace_id)
        if not trace:
            return None

        checkpoints = await self.list_checkpoints(trace_id)
        if not checkpoints:
            return None

        # Find target span index
        target_idx = None
        for i, span in enumerate(trace.spans):
            if span.span_id == target_span_id:
                target_idx = i
                break

        if target_idx is None:
            return None

        # Find nearest checkpoint before target
        nearest = None
        nearest_idx = -1

        for checkpoint in checkpoints:
            for i, span in enumerate(trace.spans):
                if span.span_id == checkpoint.span_id and i < target_idx:
                    if i > nearest_idx:
                        nearest = checkpoint
                        nearest_idx = i

        return nearest

    async def compare_checkpoints(
        self,
        checkpoint_a_id: str,
        checkpoint_b_id: str,
    ) -> Dict[str, Any]:
        """
        Compare two checkpoints.

        Args:
            checkpoint_a_id: First checkpoint ID
            checkpoint_b_id: Second checkpoint ID

        Returns:
            Comparison dictionary
        """
        cp_a = await self.get_checkpoint(checkpoint_a_id)
        cp_b = await self.get_checkpoint(checkpoint_b_id)

        if not cp_a or not cp_b:
            return {"error": "Checkpoint not found"}

        # Compare states
        state_a = cp_a.state_snapshot
        state_b = cp_b.state_snapshot

        differences = []

        # Compare span context
        ctx_a = state_a.get("span_context", {})
        ctx_b = state_b.get("span_context", {})

        if ctx_a.get("input_data") != ctx_b.get("input_data"):
            differences.append("input_data differs")

        if ctx_a.get("output_data") != ctx_b.get("output_data"):
            differences.append("output_data differs")

        # Compare custom state keys
        all_keys = set(state_a.keys()) | set(state_b.keys())
        for key in all_keys:
            if key in ("span_context", "parent_chain"):
                continue
            if state_a.get(key) != state_b.get(key):
                differences.append(f"{key} differs")

        return {
            "checkpoint_a": cp_a.to_dict(),
            "checkpoint_b": cp_b.to_dict(),
            "differences": differences,
            "same": len(differences) == 0,
        }
