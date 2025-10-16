"""LLM provider abstractions."""

from __future__ import annotations

import random
from typing import Protocol

from .context import DataContext


class LLMProvider(Protocol):
    """Lightweight protocol for text completion."""

    def complete(self, system: str, user: str, seed: int) -> str:
        ...


class TemplateProvider:
    """Template-based provider that does not call external APIs."""

    def __init__(self, ctx: DataContext):
        self.ctx = ctx

    def complete(self, system: str, user: str, seed: int) -> str:
        rng = random.Random(seed)
        templates = [
            "{user_text}",
            "{user_text} Let's ensure we account for recent quality signals.",
            "Quick take: {user_text}",
        ]
        return templates[rng.randrange(len(templates))].format(user_text=user.strip())


class OpenAILLMProvider:
    """Stub for OpenAI style provider.

    This class exists as a placeholder and is not wired by default.
    Users can extend it to call their own LLM endpoints.
    """

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key
        self.model = model

    def complete(self, system: str, user: str, seed: int) -> str:
        raise NotImplementedError(
            "OpenAILLMProvider is a stub. Provide your own implementation to call the API."
        )
