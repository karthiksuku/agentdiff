"""
Capture hooks for various LLM providers and frameworks.
"""

from .openai_hook import OpenAIHook, patch_openai
from .anthropic_hook import AnthropicHook, patch_anthropic
from .langchain_hook import LangChainCallback
from .crewai_hook import CrewAICallback
from .generic_hook import GenericLLMHook

__all__ = [
    "OpenAIHook",
    "patch_openai",
    "AnthropicHook",
    "patch_anthropic",
    "LangChainCallback",
    "CrewAICallback",
    "GenericLLMHook",
]
