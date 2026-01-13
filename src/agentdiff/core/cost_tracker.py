"""
Cost tracking for LLM API calls.

Provides pricing information for various models and calculates costs
based on token usage.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime


@dataclass
class ModelPricing:
    """Pricing information for a specific model."""
    model_id: str
    provider: str
    input_price_per_1k: float  # USD per 1000 input tokens
    output_price_per_1k: float  # USD per 1000 output tokens
    cached_input_price_per_1k: Optional[float] = None  # For cached/prompt caching
    effective_date: Optional[datetime] = None

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """
        Calculate total cost for token usage.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached input tokens

        Returns:
            Total cost in USD
        """
        # Calculate regular input cost
        regular_input_tokens = input_tokens - cached_tokens
        input_cost = (regular_input_tokens / 1000) * self.input_price_per_1k

        # Calculate cached input cost if applicable
        cached_cost = 0.0
        if cached_tokens > 0 and self.cached_input_price_per_1k is not None:
            cached_cost = (cached_tokens / 1000) * self.cached_input_price_per_1k

        # Calculate output cost
        output_cost = (output_tokens / 1000) * self.output_price_per_1k

        return input_cost + cached_cost + output_cost


class CostTracker:
    """
    Tracks costs for LLM API calls across different providers.

    Provides pricing lookup and cost calculation for various models.
    """

    # Default pricing (as of late 2024 - should be updated regularly)
    DEFAULT_PRICING: Dict[str, ModelPricing] = {
        # OpenAI models
        "gpt-4o": ModelPricing(
            model_id="gpt-4o",
            provider="openai",
            input_price_per_1k=0.0025,
            output_price_per_1k=0.01,
            cached_input_price_per_1k=0.00125,
        ),
        "gpt-4o-mini": ModelPricing(
            model_id="gpt-4o-mini",
            provider="openai",
            input_price_per_1k=0.00015,
            output_price_per_1k=0.0006,
            cached_input_price_per_1k=0.000075,
        ),
        "gpt-4-turbo": ModelPricing(
            model_id="gpt-4-turbo",
            provider="openai",
            input_price_per_1k=0.01,
            output_price_per_1k=0.03,
        ),
        "gpt-4": ModelPricing(
            model_id="gpt-4",
            provider="openai",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
        ),
        "gpt-3.5-turbo": ModelPricing(
            model_id="gpt-3.5-turbo",
            provider="openai",
            input_price_per_1k=0.0005,
            output_price_per_1k=0.0015,
        ),
        "o1-preview": ModelPricing(
            model_id="o1-preview",
            provider="openai",
            input_price_per_1k=0.015,
            output_price_per_1k=0.06,
        ),
        "o1-mini": ModelPricing(
            model_id="o1-mini",
            provider="openai",
            input_price_per_1k=0.003,
            output_price_per_1k=0.012,
        ),

        # Anthropic models
        "claude-3-5-sonnet-20241022": ModelPricing(
            model_id="claude-3-5-sonnet-20241022",
            provider="anthropic",
            input_price_per_1k=0.003,
            output_price_per_1k=0.015,
            cached_input_price_per_1k=0.0003,
        ),
        "claude-3-5-haiku-20241022": ModelPricing(
            model_id="claude-3-5-haiku-20241022",
            provider="anthropic",
            input_price_per_1k=0.0008,
            output_price_per_1k=0.004,
            cached_input_price_per_1k=0.00008,
        ),
        "claude-3-opus-20240229": ModelPricing(
            model_id="claude-3-opus-20240229",
            provider="anthropic",
            input_price_per_1k=0.015,
            output_price_per_1k=0.075,
            cached_input_price_per_1k=0.0015,
        ),
        "claude-3-sonnet-20240229": ModelPricing(
            model_id="claude-3-sonnet-20240229",
            provider="anthropic",
            input_price_per_1k=0.003,
            output_price_per_1k=0.015,
        ),
        "claude-3-haiku-20240307": ModelPricing(
            model_id="claude-3-haiku-20240307",
            provider="anthropic",
            input_price_per_1k=0.00025,
            output_price_per_1k=0.00125,
        ),

        # Google models
        "gemini-1.5-pro": ModelPricing(
            model_id="gemini-1.5-pro",
            provider="google",
            input_price_per_1k=0.00125,
            output_price_per_1k=0.005,
        ),
        "gemini-1.5-flash": ModelPricing(
            model_id="gemini-1.5-flash",
            provider="google",
            input_price_per_1k=0.000075,
            output_price_per_1k=0.0003,
        ),
        "gemini-2.0-flash": ModelPricing(
            model_id="gemini-2.0-flash",
            provider="google",
            input_price_per_1k=0.0001,
            output_price_per_1k=0.0004,
        ),

        # Mistral models
        "mistral-large": ModelPricing(
            model_id="mistral-large",
            provider="mistral",
            input_price_per_1k=0.002,
            output_price_per_1k=0.006,
        ),
        "mistral-medium": ModelPricing(
            model_id="mistral-medium",
            provider="mistral",
            input_price_per_1k=0.00275,
            output_price_per_1k=0.0081,
        ),
        "mistral-small": ModelPricing(
            model_id="mistral-small",
            provider="mistral",
            input_price_per_1k=0.0002,
            output_price_per_1k=0.0006,
        ),

        # Cohere models
        "command-r-plus": ModelPricing(
            model_id="command-r-plus",
            provider="cohere",
            input_price_per_1k=0.0025,
            output_price_per_1k=0.01,
        ),
        "command-r": ModelPricing(
            model_id="command-r",
            provider="cohere",
            input_price_per_1k=0.00015,
            output_price_per_1k=0.0006,
        ),
    }

    def __init__(self, custom_pricing: Optional[Dict[str, ModelPricing]] = None):
        """
        Initialize cost tracker.

        Args:
            custom_pricing: Optional dictionary of custom model pricing
        """
        self.pricing = dict(self.DEFAULT_PRICING)
        if custom_pricing:
            self.pricing.update(custom_pricing)

    def get_pricing(self, model: str) -> Optional[ModelPricing]:
        """
        Get pricing for a model.

        Handles model name variations (e.g., with/without version suffix).

        Args:
            model: Model identifier

        Returns:
            ModelPricing if found, None otherwise
        """
        # Direct lookup
        if model in self.pricing:
            return self.pricing[model]

        # Try without date suffix
        base_model = model.rsplit("-", 1)[0] if "-20" in model else model
        if base_model in self.pricing:
            return self.pricing[base_model]

        # Try prefix matching for versioned models
        for key in self.pricing:
            if model.startswith(key) or key.startswith(model):
                return self.pricing[key]

        return None

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """
        Calculate cost for a model call.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached input tokens

        Returns:
            Cost in USD (0.0 if model not found)
        """
        pricing = self.get_pricing(model)
        if not pricing:
            return 0.0

        return pricing.calculate_cost(input_tokens, output_tokens, cached_tokens)

    def add_pricing(self, pricing: ModelPricing):
        """Add or update pricing for a model."""
        self.pricing[pricing.model_id] = pricing

    def get_all_models(self) -> list:
        """Get list of all known model IDs."""
        return list(self.pricing.keys())

    def get_models_by_provider(self, provider: str) -> list:
        """Get all models for a specific provider."""
        return [
            model_id
            for model_id, pricing in self.pricing.items()
            if pricing.provider == provider
        ]


# Global cost tracker instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get or create the global cost tracker."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> float:
    """
    Convenience function to calculate cost.

    Args:
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cached_tokens: Number of cached input tokens

    Returns:
        Cost in USD
    """
    return get_cost_tracker().calculate_cost(
        model, input_tokens, output_tokens, cached_tokens
    )
