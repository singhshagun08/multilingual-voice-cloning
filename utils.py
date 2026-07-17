"""Shared filesystem, audio and logging utilities."""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Iterable

import librosa
import soundfile as sf

from config import SETTINGS


def get_logger(name: str) -> logging.Logger:
    """Return a file-and-console logger without adding duplicate handlers."""
    SETTINGS.create_directories()
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(SETTINGS.log_dir / "application.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def safe_stem(text: str) -> str:
    """Make a stable, safe filename fragment from an ISO language code or label."""
    return re.sub(r"[^a-z0-9_-]+", "_", text.lower()).strip("_") or "audio"


def output_path(language: str, model: str) -> Path:
    """Create a collision-safe output WAV path under ``outputs/``."""
    SETTINGS.create_directories()
    return SETTINGS.output_dir / f"{safe_stem(language)}_{safe_stem(model)}_{uuid.uuid4().hex[:8]}.wav"


def audio_duration(path: str | Path) -> float:
    """Return WAV/audio duration in seconds."""
    return float(librosa.get_duration(path=str(path)))


def normalise_audio(source: str | Path, destination: str | Path, sample_rate: int = 24000) -> Path:
    """Convert an arbitrary reference clip to mono WAV through ffmpeg."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-i", str(source), "-ac", "1", "-ar", str(sample_rate), str(destination)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr[-1000:]}")
    return destination


def concatenate_references(references: Iterable[str | Path], destination: str | Path) -> Path:
    """Concatenate normalised references using librosa; requires at least one clip."""
    clips = list(references)
    if not clips:
        raise ValueError("At least one reference clip is required for voice cloning.")
    import numpy as np
    arrays = [librosa.load(str(clip), sr=SETTINGS.sample_rate, mono=True)[0] for clip in clips]
    sf.write(str(destination), np.concatenate(arrays), SETTINGS.sample_rate)
    return Path(destination)


def python_command() -> list[str]:
    """Return the interpreter command used by the current virtual environment."""
    return [sys.executable]
