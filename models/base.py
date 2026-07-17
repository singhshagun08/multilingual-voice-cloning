"""Typed interface and timing logic shared by all speech engines."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from utils import audio_duration


@dataclass(frozen=True)
class TTSResult:
    """Output asset and performance measures emitted by an engine."""
    path: Path
    model: str
    language: str
    generation_seconds: float
    audio_seconds: float

    @property
    def rtf(self) -> float:
        """Return real-time factor; lower than one is faster than real time."""
        return self.generation_seconds / self.audio_seconds if self.audio_seconds else 0.0


class BaseTTSModel(ABC):
    """All TTS wrappers implement model loading and a concrete synthesis call."""
    name: str
    supports_voice_cloning: bool = False
    supported_languages: set[str] = set()

    def __init__(self, device: str = "cuda") -> None:
        self.device, self.loaded = device, False

    @abstractmethod
    def load(self) -> None:
        """Download/load the model once."""

    @abstractmethod
    def _synthesise(self, text: str, language: str, references: Sequence[str | Path], output: Path) -> None:
        """Write audio to output. Implementations validate language and references."""

    def generate(self, text: str, language: str, references: Sequence[str | Path], output: Path) -> TTSResult:
        """Generate WAV and record the wall-clock latency and audio duration."""
        if language not in self.supported_languages:
            raise ValueError(f"{self.name} does not support '{language}'.")
        if not text.strip():
            raise ValueError("Text cannot be empty.")
        if not self.loaded:
            self.load()
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        self._synthesise(text, language, references, output)
        generation = time.perf_counter() - started
        return TTSResult(output, self.name, language, generation, audio_duration(output))
