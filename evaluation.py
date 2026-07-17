"""ASR, error-rate and speaker-similarity evaluation for generated audio."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from jiwer import cer, wer
from speechbrain.inference.speaker import EncoderClassifier

from config import SETTINGS
from utils import audio_duration, get_logger

LOGGER = get_logger(__name__)


def normalise_transcript(text: str) -> str:
    """Normalize text consistently before WER/CER scoring."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\u0600-\u06ff]", "", text.lower())).strip()


class ASRTranscriber:
    """Lazy ASR abstraction for Whisper, faster-whisper and IndicWhisper."""

    def __init__(self, engine: Literal["whisper", "faster-whisper", "indicwhisper"] = "faster-whisper", device: str | None = None) -> None:
        self.engine, self.device, self.model = engine, (device or SETTINGS.device), None

    def _load(self) -> None:
        if self.engine == "faster-whisper":
            from faster_whisper import WhisperModel
            self.model = WhisperModel("large-v3", device="cuda" if self.device.startswith("cuda") else "cpu", compute_type="float16" if self.device.startswith("cuda") else "int8")
        else:
            from transformers import pipeline
            model = SETTINGS.whisper_model if self.engine == "whisper" else SETTINGS.indic_whisper_model
            self.model = pipeline("automatic-speech-recognition", model=model, device=0 if self.device.startswith("cuda") else -1)

    def transcribe(self, audio: str | Path, language: str) -> tuple[str, float]:
        """Transcribe an audio file and return text plus ASR execution time."""
        if self.model is None:
            self._load()
        started = time.perf_counter()
        if self.engine == "faster-whisper":
            segments, _ = self.model.transcribe(str(audio), language=language, beam_size=5)
            transcript = "".join(segment.text for segment in segments).strip()
        else:
            result = self.model(str(audio), generate_kwargs={"language": language, "task": "transcribe"})
            transcript = result["text"].strip()
        elapsed = time.perf_counter() - started
        LOGGER.info("ASR engine=%s audio=%s latency=%.2fs", self.engine, audio, elapsed)
        return transcript, elapsed


class SpeakerSimilarity:
    """ECAPA-TDNN cosine speaker similarity scorer."""

    def __init__(self, device: str | None = None) -> None:
        self.device, self.classifier = device or SETTINGS.device, None

    def _load(self) -> None:
        """Download/load SpeechBrain's pretrained ECAPA embedding model."""
        run_opts = {"device": self.device if torch.cuda.is_available() else "cpu"}
        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts=run_opts)

    def _load_wav_16k(self, path: str | Path):
        """Load audio via torchaudio directly, bypassing SpeechBrain's load_audio
        (which triggers a broken lazy import of the optional k2_fsa integration module —
        see README section 6 for details on this known SpeechBrain issue)."""
        import torchaudio
        signal, sr = torchaudio.load(str(path))
        if sr != 16000:
            signal = torchaudio.functional.resample(signal, sr, 16000)
        return signal.mean(dim=0)

    def score(self, reference: str | Path, generated: str | Path) -> float:
        """Return cosine similarity of two speaker embeddings in [-1, 1]."""
        if self.classifier is None:
            self._load()
        first = self.classifier.encode_batch(self._load_wav_16k(reference).unsqueeze(0)).squeeze()
        second = self.classifier.encode_batch(self._load_wav_16k(generated).unsqueeze(0)).squeeze()
        return float(torch.nn.functional.cosine_similarity(first.unsqueeze(0), second.unsqueeze(0)).item())


@dataclass(frozen=True)
class EvaluationResult:
    """All quality metrics stored in benchmark rows."""
    transcript: str
    wer: float
    cer: float
    speaker_similarity: float | None
    asr_seconds: float
    mos: str = "Not available: requires human listening study"


def evaluate_audio(audio: str | Path, expected_text: str, language: str, reference_audio: str | Path | None = None, asr_engine: str = "faster-whisper") -> EvaluationResult:
    """Transcribe audio, calculate WER/CER, and optionally score cloned speaker similarity."""
    transcriber = ASRTranscriber(asr_engine)  # type: ignore[arg-type]
    transcript, asr_seconds = transcriber.transcribe(audio, language)
    expected, actual = normalise_transcript(expected_text), normalise_transcript(transcript)
    similarity = SpeakerSimilarity().score(reference_audio, audio) if reference_audio else None
    result = EvaluationResult(transcript, float(wer(expected, actual)), float(cer(expected, actual)), similarity, asr_seconds)
    LOGGER.info("Evaluation audio=%s WER=%.3f CER=%.3f similarity=%s", audio, result.wer, result.cer, result.speaker_similarity)
    return result
