"""
Anthropic API interceptor for automatic tracing.
"""

from typing import Optional, Any, Dict
from functools import wraps
import logging

from ..core.tracer import get_current_span
from ..core.span import SpanType, TokenUsage
from ..core.cost_tracker import get_cost_tracker

logger = logging.getLogger(__name__)


class AnthropicHook:
    """
    Hook for intercepting Anthropic API calls.

    Usage:
        from agentdiff.capture import AnthropicHook

        hook = AnthropicHook()
        hook.install()

        # Now all Anthropic calls will be traced
        client = Anthropic()
        response = client.messages.create(...)
    """

    def __init__(self, auto_span: bool = True):
        """
        Initialize the hook.

        Args:
            auto_span: Whether to automatically create spans for calls
        """
        self.auto_span = auto_span
        self._original_create = None
        self._original_async_create = None
        self._installed = False

    def install(self) -> None:
        """Install the hook by patching Anthropic."""
        if self._installed:
            return

        try:
            import anthropic
            from anthropic.resources import messages

            # Patch sync create
            self._original_create = messages.Messages.create

            @wraps(self._original_create)
            def patched_create(self_client, *args, **kwargs):
                return self._wrap_call(
                    self._original_create,
                    self_client,
                    *args,
                    **kwargs
                )

            messages.Messages.create = patched_create

            # Patch async create
            self._original_async_create = messages.AsyncMessages.create

            @wraps(self._original_async_create)
            async def patched_async_create(self_client, *args, **kwargs):
                return await self._wrap_async_call(
                    self._original_async_create,
                    self_client,
                    *args,
                    **kwargs
                )

            messages.AsyncMessages.create = patched_async_create

            self._installed = True
            logger.info("Anthropic hook installed")

        except ImportError:
            logger.warning("Anthropic package not installed, hook not available")

    def uninstall(self) -> None:
        """Uninstall the hook."""
        if not self._installed:
            return

        try:
            from anthropic.resources import messages

            if self._original_create:
                messages.Messages.create = self._original_create
            if self._original_async_create:
                messages.AsyncMessages.create = self._original_async_create

            self._installed = False
            logger.info("Anthropic hook uninstalled")

        except ImportError:
            pass

    def _wrap_call(self, original_fn, client, *args, **kwargs):
        """Wrap a synchronous Anthropic call."""
        span = get_current_span()

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system")

        try:
            response = original_fn(client, *args, **kwargs)

            if span:
                self._log_to_span(span, model, messages, system, response)

            return response

        except Exception as e:
            if span:
                span.error = str(e)
                span.error_type = type(e).__name__
            raise

    async def _wrap_async_call(self, original_fn, client, *args, **kwargs):
        """Wrap an asynchronous Anthropic call."""
        span = get_current_span()

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system")

        try:
            response = await original_fn(client, *args, **kwargs)

            if span:
                self._log_to_span(span, model, messages, system, response)

            return response

        except Exception as e:
            if span:
                span.error = str(e)
                span.error_type = type(e).__name__
            raise

    def _log_to_span(self, span, model: str, messages: list, system: Optional[str], response) -> None:
        """Log Anthropic response to span."""
        span.model = model
        span.provider = "anthropic"
        span.span_type = SpanType.LLM_CALL

        input_data = {"messages": messages}
        if system:
            input_data["system"] = system
        span.input_data = input_data

        # Extract response content
        if hasattr(response, "content") and response.content:
            content_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    content_parts.append(block.text)

            span.output_data = {
                "response": "\n".join(content_parts),
                "stop_reason": response.stop_reason,
                "role": response.role,
            }

        # Extract token usage
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            cached_tokens = 0

            # Handle cache tokens if available
            if hasattr(usage, "cache_creation_input_tokens"):
                cached_tokens = getattr(usage, "cache_read_input_tokens", 0)

            span.token_usage = TokenUsage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
                cached_tokens=cached_tokens,
            )

            # Calculate cost
            cost_tracker = get_cost_tracker()
            pricing = cost_tracker.get_pricing(model)
            if pricing:
                span.token_usage.calculate_cost(
                    pricing.input_price_per_1k,
                    pricing.output_price_per_1k,
                )


def patch_anthropic() -> AnthropicHook:
    """
    Convenience function to patch Anthropic.

    Returns:
        Installed AnthropicHook
    """
    hook = AnthropicHook()
    hook.install()
    return hook
