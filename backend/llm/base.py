"""Provider-agnostic LLM interface.

Switch providers via LLM_PROVIDER env var: "watsonx" | "custom" | "mock".
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from backend.config import CONFIG


class LLM(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, max_tokens: int = 512,
                 temperature: float = 0.2) -> str:
        ...


def get_llm() -> LLM:
    provider = CONFIG.llm_provider
    if provider == "watsonx":
        from .watsonx_provider import WatsonxLLM
        return WatsonxLLM()
    if provider == "custom":
        from .custom_provider import CustomLLM
        return CustomLLM()
    from .mock_provider import MockLLM
    return MockLLM()
