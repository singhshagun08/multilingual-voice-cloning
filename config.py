"""Central, environment-variable based configuration for the application."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    """Runtime settings. Override values with ``MVC_*`` environment variables."""

    device: str = field(default_factory=lambda: os.getenv("MVC_DEVICE", "cuda"))
    output_dir: Path = ROOT / "outputs"
    benchmark_dir: Path = ROOT / "benchmark"
    log_dir: Path = ROOT / "logs"
    sample_rate: int = int(os.getenv("MVC_SAMPLE_RATE", "24000"))
    nllb_model: str = os.getenv("MVC_NLLB_MODEL", "facebook/nllb-200-distilled-600M")
    seamless_model: str = os.getenv("MVC_SEAMLESS_MODEL", "facebook/seamless-m4t-v2-large")
    whisper_model: str = os.getenv("MVC_WHISPER_MODEL", "openai/whisper-large-v3")
    # NOTE: "ai4bharat/indicwhisper" does not exist on Hugging Face (confirmed 404).
    # The working Hindi-tuned Whisper fine-tune used and benchmarked in this project is
    # vasista22/whisper-hindi-small (see README section 5, Finding #1: WER dropped from
    # 0.417 with generic faster-whisper to 0.167 with this model on the same audio clip).
    indic_whisper_model: str = os.getenv("MVC_INDIC_WHISPER_MODEL", "vasista22/whisper-hindi-small")
    # English defaults to Chatterbox (best measured similarity/MOS in benchmarks — see README section 3).
    # Hindi/Arabic default to XTTS since Chatterbox has no multilingual support.
    router: Mapping[str, str] = field(default_factory=lambda: {
        "en": os.getenv("MVC_EN_MODEL", "chatterbox"),
        "hi": os.getenv("MVC_HI_MODEL", "xtts"),
        "ar": os.getenv("MVC_AR_MODEL", "xtts"),
    })

    def create_directories(self) -> None:
        """Create all runtime directories before models write files."""
        for path in (self.output_dir, self.benchmark_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
