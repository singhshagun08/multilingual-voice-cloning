"""ResembleAI Chatterbox open-source voice cloning wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torchaudio

from .base import BaseTTSModel


class ChatterboxModel(BaseTTSModel):
    """Chatterbox supports English cloning; use XTTS for Hindi/Arabic routing."""
    name = "chatterbox"
    supports_voice_cloning = True
    supported_languages = {"en"}

    def load(self) -> None:
        """Download/cache Chatterbox using its official Python API."""
        from chatterbox.tts import ChatterboxTTS
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.loaded = True

    def _synthesise(self, text: str, language: str, references: Sequence[str | Path], output: Path) -> None:
        """Generate English audio using the first user reference clip."""
        if not references:
            raise ValueError("Chatterbox requires a reference clip.")
        audio = self.model.generate(text, audio_prompt_path=str(references[0]))
        torchaudio.save(str(output), audio.detach().cpu(), self.model.sr)
