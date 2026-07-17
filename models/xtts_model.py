"""Coqui XTTS-v2 multilingual zero-shot voice-cloning wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from TTS.api import TTS

from .base import BaseTTSModel


class XTTSModel(BaseTTSModel):
    """XTTS-v2 supports zero-shot cloning in English, Hindi and Arabic."""
    name = "xtts"
    supports_voice_cloning = True
    supported_languages = {"en", "hi", "ar"}
    _language_codes = {"en": "en", "hi": "hi", "ar": "ar"}

    def load(self) -> None:
        """Download/cache XTTS-v2 weights through Coqui's model manager."""
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=self.device.startswith("cuda"))
        self.loaded = True

    def _synthesise(self, text: str, language: str, references: Sequence[str | Path], output: Path) -> None:
        """Synthesize from one or multiple reference clips supplied to XTTS."""
        if not references:
            raise ValueError("XTTS requires one or more reference clips.")
        self.tts.tts_to_file(text=text, speaker_wav=[str(p) for p in references], language=self._language_codes[language], file_path=str(output))
