"""
LangChain callback handler for automatic tracing.
"""

from typing import Any, Dict, List, Optional, Union
from uuid import UUID
import logging

from ..core.tracer import get_current_trace, get_current_span
from ..core.span import Span, SpanType, SpanStatus, TokenUsage

logger = logging.getLogger(__name__)


class LangChainCallback:
    """
    LangChain callback handler for AgentDiff tracing.

    Usage:
        from agentdiff.capture import LangChainCallback
        from langchain.chat_models import ChatOpenAI

        callback = LangChainCallback()

        llm = ChatOpenAI(callbacks=[callback])
        # or
        chain.invoke(input, config={"callbacks": [callback]})
    """

    def __init__(self):
        """Initialize the callback handler."""
        self._run_spans: Dict[str, Span] = {}

    def _get_or_create_span(
        self,
        run_id: UUID,
        name: str,
        span_type: SpanType,
    ) -> Span:
        """Get or create a span for a run."""
        run_id_str = str(run_id)

        if run_id_str in self._run_spans:
            return self._run_spans[run_id_str]

        trace = get_current_trace()
        parent_span = get_current_span()

        span = Span(
            trace_id=trace.trace_id if trace else "",
            parent_span_id=parent_span.span_id if parent_span else None,
            name=name,
            span_type=span_type,
        )

        if trace:
            trace.add_span(span)

        self._run_spans[run_id_str] = span
        return span

    def _finish_span(self, run_id: UUID, status: SpanStatus = SpanStatus.COMPLETED) -> Optional[Span]:
        """Finish and clean up a span."""
        run_id_str = str(run_id)
        span = self._run_spans.pop(run_id_str, None)
        if span:
            span.finish(status)
        return span

    # LLM callbacks
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM starts running."""
        model_name = serialized.get("name", "unknown")
        span = self._get_or_create_span(run_id, f"llm:{model_name}", SpanType.LLM_CALL)
        span.model = model_name
        span.provider = serialized.get("_type", "langchain")
        span.input_data = {"prompts": prompts}
        if tags:
            span.tags = tags
        if metadata:
            span.metadata.update(metadata)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM ends running."""
        span = self._run_spans.get(str(run_id))
        if not span:
            return

        # Extract generations
        if hasattr(response, "generations") and response.generations:
            outputs = []
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "text"):
                        outputs.append(gen.text)
            span.output_data = {"generations": outputs}

        # Extract token usage
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                span.token_usage = TokenUsage(
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )

        self._finish_span(run_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when LLM errors."""
        span = self._run_spans.get(str(run_id))
        if span:
            span.error = str(error)
            span.error_type = type(error).__name__
        self._finish_span(run_id, SpanStatus.FAILED)

    # Chat model callbacks
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chat model starts."""
        model_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        span = self._get_or_create_span(run_id, f"chat:{model_name}", SpanType.LLM_CALL)
        span.model = model_name
        span.provider = "langchain"

        # Convert messages to serializable format
        serialized_messages = []
        for msg_list in messages:
            for msg in msg_list:
                if hasattr(msg, "content"):
                    serialized_messages.append({
                        "role": getattr(msg, "type", "unknown"),
                        "content": msg.content,
                    })

        span.input_data = {"messages": serialized_messages}

        if tags:
            span.tags = tags
        if metadata:
            span.metadata.update(metadata)

    # Chain callbacks
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain starts."""
        chain_name = serialized.get("name", serialized.get("id", ["unknown"])[-1])
        span = self._get_or_create_span(run_id, f"chain:{chain_name}", SpanType.AGENT_STEP)
        span.input_data = inputs
        if tags:
            span.tags = tags
        if metadata:
            span.metadata.update(metadata)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain ends."""
        span = self._run_spans.get(str(run_id))
        if span:
            span.output_data = outputs
        self._finish_span(run_id)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when chain errors."""
        span = self._run_spans.get(str(run_id))
        if span:
            span.error = str(error)
            span.error_type = type(error).__name__
        self._finish_span(run_id, SpanStatus.FAILED)

    # Tool callbacks
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool starts."""
        tool_name = serialized.get("name", "unknown")
        span = self._get_or_create_span(run_id, f"tool:{tool_name}", SpanType.TOOL_CALL)
        span.input_data = {"tool": tool_name, "input": input_str}
        if tags:
            span.tags = tags
        if metadata:
            span.metadata.update(metadata)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool ends."""
        span = self._run_spans.get(str(run_id))
        if span:
            span.output_data = {"output": output}
        self._finish_span(run_id)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when tool errors."""
        span = self._run_spans.get(str(run_id))
        if span:
            span.error = str(error)
            span.error_type = type(error).__name__
        self._finish_span(run_id, SpanStatus.FAILED)

    # Retriever callbacks
    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever starts."""
        retriever_name = serialized.get("name", "retriever")
        span = self._get_or_create_span(run_id, f"retrieval:{retriever_name}", SpanType.RETRIEVAL)
        span.input_data = {"query": query}
        if tags:
            span.tags = tags
        if metadata:
            span.metadata.update(metadata)

    def on_retriever_end(
        self,
        documents: List[Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever ends."""
        span = self._run_spans.get(str(run_id))
        if span:
            # Serialize documents
            docs = []
            for doc in documents:
                if hasattr(doc, "page_content"):
                    docs.append({
                        "content": doc.page_content[:500],  # Truncate
                        "metadata": getattr(doc, "metadata", {}),
                    })
            span.output_data = {"documents": docs, "count": len(documents)}
        self._finish_span(run_id)

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when retriever errors."""
        span = self._run_spans.get(str(run_id))
        if span:
            span.error = str(error)
            span.error_type = type(error).__name__
        self._finish_span(run_id, SpanStatus.FAILED)

    # Agent callbacks
    def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent takes an action."""
        span = self._run_spans.get(str(run_id))
        if span:
            action_data = {
                "tool": getattr(action, "tool", "unknown"),
                "tool_input": getattr(action, "tool_input", {}),
                "log": getattr(action, "log", ""),
            }
            span.metadata["action"] = action_data

    def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when agent finishes."""
        span = self._run_spans.get(str(run_id))
        if span:
            span.output_data = {
                "output": getattr(finish, "return_values", {}),
                "log": getattr(finish, "log", ""),
            }
