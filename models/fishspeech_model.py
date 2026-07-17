"""Fish Speech API wrapper for multilingual reference-conditioned generation."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from .base import BaseTTSModel

# NOTE: Never successfully installed/tested in this project.
# Fish Speech's pyproject.toml pins tokenizers<0.11 (~2021), which has
# no prebuilt wheel for Python 3.12 and fails to build from source.
# See README section 4 and 6.
class FishSpeechModel(BaseTTSModel):
    """Invoke the official Fish Speech inference CLI installed from its repository."""
    name = "fishspeech"
    supports_voice_cloning = True
    supported_languages = {"en", "hi", "ar"}
    _language_names = {"en": "English", "hi": "Hindi", "ar": "Arabic"}

    def load(self) -> None:
        """Validate that the official Fish Speech inference package is importable."""
        try:
            import fish_speech  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install Fish Speech with: pip install -e third_party/Fish-Speech") from exc
        self.loaded = True

    def _synthesise(self, text: str, language: str, references: Sequence[str | Path], output: Path) -> None:
        """Call Fish Speech's maintained batch CLI and fail with its diagnostic."""
        if not references:
            raise ValueError("Fish Speech requires a reference clip.")
        command = ["fish-speech", "tts", "--text", text, "--reference-audio", str(references[0]), "--language", self._language_names[language], "--output", str(output)]
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode:
            raise RuntimeError(f"Fish Speech failed: {process.stderr[-2000:]}")
