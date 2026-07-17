"""Configurable language-to-TTS model routing."""
from __future__ import annotations

from dataclasses import dataclass

from config import SETTINGS
from utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class LanguageRouter:
    """Select a registered TTS model by language with explicit user overrides."""
    routes: dict[str, str]

    @classmethod
    def default(cls) -> "LanguageRouter":
        """Build router from central settings and environment overrides."""
        return cls(dict(SETTINGS.router))

    def choose(self, language: str, requested_model: str | None = None) -> str:
        """Return requested model or configured best model for the language."""
        model = requested_model or self.routes.get(language)
        if not model:
            raise ValueError(f"No TTS route configured for language '{language}'.")
        LOGGER.info("Model router language=%s selected=%s", language, model)
        return model
