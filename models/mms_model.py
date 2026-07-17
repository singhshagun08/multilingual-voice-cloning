"""Meta MMS-TTS wrapper (multilingual TTS baseline; not a cloning model)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import soundfile as sf
import torch
from transformers import VitsModel, AutoTokenizer

from .base import BaseTTSModel


class MMSModel(BaseTTSModel):
    """MMS provides a no-cloning multilingual benchmark baseline."""
    name = "mms"
    supports_voice_cloning = False
    supported_languages = {"en", "hi", "ar"}
    _models = {"en": "facebook/mms-tts-eng", "hi": "facebook/mms-tts-hin", "ar": "facebook/mms-tts-ara"}

    def load(self) -> None:
        """Defer language-specific loading until generation."""
        self.model = self.tokenizer = self.current_language = None
        self.loaded = True

    def _synthesise(self, text: str, language: str, references: Sequence[str | Path], output: Path) -> None:
        """Synthesize with the language-specific MMS VITS checkpoint."""
        if self.current_language != language:
            self.tokenizer = AutoTokenizer.from_pretrained(self._models[language])
            self.model = VitsModel.from_pretrained(self._models[language]).to(self.device)
            self.current_language = language
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            waveform = self.model(**inputs).waveform[0].cpu().numpy()
        sf.write(str(output), waveform, self.model.config.sampling_rate)
