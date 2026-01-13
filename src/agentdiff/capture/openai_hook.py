"""
OpenAI API interceptor for automatic tracing.
"""

from typing import Optional, Any, Dict
from functools import wraps
import logging

from ..core.tracer import get_current_span, get_tracer
from ..core.span import SpanType, TokenUsage
from ..core.cost_tracker import get_cost_tracker

logger = logging.getLogger(__name__)


class OpenAIHook:
    """
    Hook for intercepting OpenAI API calls.

    Usage:
        from agentdiff.capture import OpenAIHook

        hook = OpenAIHook()
        hook.install()

        # Now all OpenAI calls will be traced
        client = OpenAI()
        response = client.chat.completions.create(...)
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
        """Install the hook by patching OpenAI."""
        if self._installed:
            return

        try:
            import openai
            from openai.resources.chat import completions

            # Patch sync create
            self._original_create = completions.Completions.create

            @wraps(self._original_create)
            def patched_create(self_client, *args, **kwargs):
                return self._wrap_call(
                    self._original_create,
                    self_client,
                    *args,
                    **kwargs
                )

            completions.Completions.create = patched_create

            # Patch async create
            self._original_async_create = completions.AsyncCompletions.create

            @wraps(self._original_async_create)
            async def patched_async_create(self_client, *args, **kwargs):
                return await self._wrap_async_call(
                    self._original_async_create,
                    self_client,
                    *args,
                    **kwargs
                )

            completions.AsyncCompletions.create = patched_async_create

            self._installed = True
            logger.info("OpenAI hook installed")

        except ImportError:
            logger.warning("OpenAI package not installed, hook not available")

    def uninstall(self) -> None:
        """Uninstall the hook."""
        if not self._installed:
            return

        try:
            from openai.resources.chat import completions

            if self._original_create:
                completions.Completions.create = self._original_create
            if self._original_async_create:
                completions.AsyncCompletions.create = self._original_async_create

            self._installed = False
            logger.info("OpenAI hook uninstalled")

        except ImportError:
            pass

    def _wrap_call(self, original_fn, client, *args, **kwargs):
        """Wrap a synchronous OpenAI call."""
        span = get_current_span()

        # Extract parameters
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])

        try:
            # Make the actual call
            response = original_fn(client, *args, **kwargs)

            # Log to span if available
            if span:
                self._log_to_span(span, model, messages, response)

            return response

        except Exception as e:
            if span:
                span.error = str(e)
                span.error_type = type(e).__name__
            raise

    async def _wrap_async_call(self, original_fn, client, *args, **kwargs):
        """Wrap an asynchronous OpenAI call."""
        span = get_current_span()

        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])

        try:
            response = await original_fn(client, *args, **kwargs)

            if span:
                self._log_to_span(span, model, messages, response)

            return response

        except Exception as e:
            if span:
                span.error = str(e)
                span.error_type = type(e).__name__
            raise

    def _log_to_span(self, span, model: str, messages: list, response) -> None:
        """Log OpenAI response to span."""
        span.model = model
        span.provider = "openai"
        span.span_type = SpanType.LLM_CALL

        span.input_data = {"messages": messages}

        # Extract response content
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                span.output_data = {
                    "response": choice.message.content,
                    "role": choice.message.role,
                    "finish_reason": choice.finish_reason,
                }

        # Extract token usage
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            span.token_usage = TokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cached_tokens=getattr(usage, "prompt_tokens_details", {}).get("cached_tokens", 0)
                if hasattr(usage, "prompt_tokens_details") else 0,
            )

            # Calculate cost
            cost_tracker = get_cost_tracker()
            pricing = cost_tracker.get_pricing(model)
            if pricing:
                span.token_usage.calculate_cost(
                    pricing.input_price_per_1k,
                    pricing.output_price_per_1k,
                )


def patch_openai() -> OpenAIHook:
    """
    Convenience function to patch OpenAI.

    Returns:
        Installed OpenAIHook
    """
    hook = OpenAIHook()
    hook.install()
    return hook
