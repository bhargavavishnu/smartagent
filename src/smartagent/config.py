"""Configuration and provider selection for SmartAgent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

_PLACEHOLDER_KEYS = {"", "your_openai_api_key_here", "sk-your-key"}


@dataclass(frozen=True)
class Settings:
    """Runtime settings for SmartAgent."""

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    mode: str = "auto"
    max_retries: int = 3
    max_tool_rounds: int = 6

    @classmethod
    def from_env(cls, openai_api_key_override: str | None = None) -> "Settings":
        key = (
            openai_api_key_override
            if openai_api_key_override is not None
            else os.getenv("OPENAI_API_KEY", "")
        ).strip()
        return cls(
            openai_api_key=key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
            mode=os.getenv("SMARTAGENT_MODE", "auto").strip().lower(),
        )

    def has_valid_openai_key(self) -> bool:
        return self.openai_api_key not in _PLACEHOLDER_KEYS

    def resolve_mode(self) -> str:
        if self.mode == "local":
            return "local"
        if self.mode == "openai":
            if not self.has_valid_openai_key():
                raise ValueError("SMARTAGENT_MODE=openai requires a valid OPENAI_API_KEY.")
            return "openai"
        return "openai" if self.has_valid_openai_key() else "local"
