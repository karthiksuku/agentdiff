"""
Semantic diff engine for comparing agent traces using embeddings.

Provides semantic similarity analysis beyond structural matching,
allowing detection of semantically equivalent but syntactically different outputs.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from ..core.span import Span
from ..core.trace import Trace


@dataclass
class SemanticSpanDiff:
    """Semantic comparison between two spans."""
    span_a: Span
    span_b: Span

    input_similarity: float = 0.0
    output_similarity: float = 0.0
    overall_similarity: float = 0.0

    # Semantic categories
    is_semantically_equivalent: bool = False
    has_semantic_drift: bool = False
    drift_direction: Optional[str] = None  # "more_specific", "more_general", "different_topic"

    # Key differences
    key_differences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "span_a_id": self.span_a.span_id,
            "span_b_id": self.span_b.span_id,
            "input_similarity": self.input_similarity,
            "output_similarity": self.output_similarity,
            "overall_similarity": self.overall_similarity,
            "is_semantically_equivalent": self.is_semantically_equivalent,
            "has_semantic_drift": self.has_semantic_drift,
            "drift_direction": self.drift_direction,
            "key_differences": self.key_differences,
        }


@dataclass
class SemanticTraceDiff:
    """Semantic comparison between two traces."""
    trace_a: Trace
    trace_b: Trace

    overall_semantic_similarity: float = 0.0
    span_diffs: List[SemanticSpanDiff] = field(default_factory=list)

    # Analysis
    semantic_equivalent_count: int = 0
    semantic_drift_count: int = 0
    major_differences: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_a_id": self.trace_a.trace_id,
            "trace_b_id": self.trace_b.trace_id,
            "overall_semantic_similarity": self.overall_semantic_similarity,
            "span_diffs": [d.to_dict() for d in self.span_diffs],
            "semantic_equivalent_count": self.semantic_equivalent_count,
            "semantic_drift_count": self.semantic_drift_count,
            "major_differences": self.major_differences,
        }


class SemanticDiffEngine:
    """
    Engine for computing semantic diffs between agent traces.

    Uses embeddings to compare the meaning of inputs and outputs,
    allowing detection of semantically similar but textually different content.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        equivalence_threshold: float = 0.9,
        drift_threshold: float = 0.7,
    ):
        """
        Initialize the semantic diff engine.

        Args:
            model_name: Sentence transformer model for embeddings
            equivalence_threshold: Similarity threshold for semantic equivalence
            drift_threshold: Threshold below which drift is detected
        """
        self.model_name = model_name
        self.equivalence_threshold = equivalence_threshold
        self.drift_threshold = drift_threshold
        self._model = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure the embedding model is loaded."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._initialized = True
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for semantic diff. "
                "Install with: pip install sentence-transformers"
            )

    def _compute_embedding(self, text: str) -> np.ndarray:
        """Compute embedding for text."""
        self._ensure_initialized()
        if not text:
            return np.zeros(384)  # Default dimension for MiniLM
        return self._model.encode(text[:512])  # Truncate long texts

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _get_embedding(self, span: Span, field: str) -> np.ndarray:
        """Get or compute embedding for a span field."""
        if field == "input" and span.input_embedding:
            return np.array(span.input_embedding)
        elif field == "output" and span.output_embedding:
            return np.array(span.output_embedding)

        # Compute from data
        if field == "input":
            text = str(span.input_data)
        else:
            text = str(span.output_data)

        return self._compute_embedding(text)

    def diff_spans(self, span_a: Span, span_b: Span) -> SemanticSpanDiff:
        """
        Compute semantic diff between two spans.

        Args:
            span_a: First span
            span_b: Second span

        Returns:
            SemanticSpanDiff with similarity analysis
        """
        # Get embeddings
        input_a = self._get_embedding(span_a, "input")
        input_b = self._get_embedding(span_b, "input")
        output_a = self._get_embedding(span_a, "output")
        output_b = self._get_embedding(span_b, "output")

        # Calculate similarities
        input_similarity = self._cosine_similarity(input_a, input_b)
        output_similarity = self._cosine_similarity(output_a, output_b)
        overall_similarity = (input_similarity + output_similarity) / 2

        # Determine semantic properties
        is_equivalent = overall_similarity >= self.equivalence_threshold
        has_drift = overall_similarity < self.drift_threshold

        # Analyze drift direction
        drift_direction = None
        if has_drift:
            drift_direction = self._analyze_drift(output_a, output_b)

        # Identify key differences
        key_differences = []
        if input_similarity < self.drift_threshold:
            key_differences.append("Input prompts differ significantly")
        if output_similarity < self.drift_threshold:
            key_differences.append("Outputs differ significantly")
        if span_a.reasoning != span_b.reasoning:
            key_differences.append("Reasoning paths differ")

        return SemanticSpanDiff(
            span_a=span_a,
            span_b=span_b,
            input_similarity=input_similarity,
            output_similarity=output_similarity,
            overall_similarity=overall_similarity,
            is_semantically_equivalent=is_equivalent,
            has_semantic_drift=has_drift,
            drift_direction=drift_direction,
            key_differences=key_differences,
        )

    def _analyze_drift(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> str:
        """
        Analyze the direction of semantic drift.

        This is a simplified heuristic. In practice, you might use
        more sophisticated methods or reference corpora.

        Args:
            embedding_a: First embedding
            embedding_b: Second embedding

        Returns:
            Drift direction description
        """
        # Simple heuristic based on embedding norms
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)

        if norm_b > norm_a * 1.1:
            return "more_specific"
        elif norm_a > norm_b * 1.1:
            return "more_general"
        else:
            return "different_topic"

    def diff(self, trace_a: Trace, trace_b: Trace) -> SemanticTraceDiff:
        """
        Compute semantic diff between two traces.

        Args:
            trace_a: First trace
            trace_b: Second trace

        Returns:
            SemanticTraceDiff with analysis
        """
        result = SemanticTraceDiff(trace_a=trace_a, trace_b=trace_b)

        # Match spans by position and type
        matched_pairs = self._match_spans(trace_a.spans, trace_b.spans)

        similarities = []

        for span_a, span_b in matched_pairs:
            span_diff = self.diff_spans(span_a, span_b)
            result.span_diffs.append(span_diff)
            similarities.append(span_diff.overall_similarity)

            if span_diff.is_semantically_equivalent:
                result.semantic_equivalent_count += 1
            if span_diff.has_semantic_drift:
                result.semantic_drift_count += 1
                if span_diff.key_differences:
                    result.major_differences.extend(span_diff.key_differences)

        # Calculate overall similarity
        if similarities:
            result.overall_semantic_similarity = float(np.mean(similarities))

        return result

    def _match_spans(
        self,
        spans_a: List[Span],
        spans_b: List[Span],
    ) -> List[Tuple[Span, Span]]:
        """
        Match spans between traces for comparison.

        Uses a simple positional matching with type filtering.

        Args:
            spans_a: Spans from first trace
            spans_b: Spans from second trace

        Returns:
            List of matched span pairs
        """
        matched = []
        used_b = set()

        for span_a in spans_a:
            best_match = None
            best_score = 0

            for i, span_b in enumerate(spans_b):
                if i in used_b:
                    continue

                # Prefer same name and type
                score = 0
                if span_a.name == span_b.name:
                    score += 2
                if span_a.span_type == span_b.span_type:
                    score += 1

                if score > best_score:
                    best_score = score
                    best_match = (i, span_b)

            if best_match:
                matched.append((span_a, best_match[1]))
                used_b.add(best_match[0])

        return matched

    def find_similar_spans(
        self,
        query_span: Span,
        candidate_spans: List[Span],
        top_k: int = 5,
    ) -> List[Tuple[Span, float]]:
        """
        Find spans most similar to a query span.

        Args:
            query_span: The span to search for
            candidate_spans: List of candidate spans
            top_k: Number of results to return

        Returns:
            List of (span, similarity) tuples
        """
        query_embedding = self._get_embedding(query_span, "output")

        similarities = []
        for span in candidate_spans:
            span_embedding = self._get_embedding(span, "output")
            similarity = self._cosine_similarity(query_embedding, span_embedding)
            similarities.append((span, similarity))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]
