"""Reproducible model, detector and translation benchmark command-line runner."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from config import SETTINGS
from evaluation import evaluate_audio
from models import MODEL_CLASSES
from translate import Translator, benchmark_detectors
from utils import get_logger, output_path

LOGGER = get_logger(__name__)
DETECTION_SAMPLES = [("Hello, this is an English sentence.", "en"), ("नमस्ते, यह हिंदी वाक्य है।", "hi"), ("مرحبا، هذه جملة عربية.", "ar")]
TRANSLATION_CASES = [("Hello, how are you?", "en", "hi"), ("नमस्ते, आप कैसे हैं?", "hi", "en"), ("كيف حالك؟", "ar", "en"), ("Good morning", "en", "ar"), ("आपका स्वागत है", "hi", "ar"), ("أهلا بك", "ar", "hi")]


def benchmark_translations() -> pd.DataFrame:
    """Run NLLB and SeamlessM4T over the required translation directions."""
    rows = []
    for engine in ("nllb", "seamless"):
        translator = Translator(engine)  # type: ignore[arg-type]
        for text, source, target in TRANSLATION_CASES:
            result = translator.translate(text, source, target)
            rows.append({"engine": engine, "source": source, "target": target, "latency_seconds": result.latency_seconds, "translation": result.text})
    return pd.DataFrame(rows)


def benchmark_tts(reference: str | Path, models: list[str], text: str, language: str, evaluate: bool = True) -> pd.DataFrame:
    """Benchmark requested TTS wrappers and persist required CSV/Markdown artifacts."""
    rows = []
    for name in models:
        model = MODEL_CLASSES[name]()
        try:
            generated = model.generate(text, language, [reference], output_path(language, name))
            quality = evaluate_audio(generated.path, text, language, reference) if evaluate else None
            rows.append({"Language": language, "Model": name, "Latency": generated.generation_seconds, "RTF": generated.rtf, "WER": quality.wer if quality else None, "CER": quality.cer if quality else None, "Similarity": quality.speaker_similarity if quality else None, "Generation Time": generated.generation_seconds, "Audio Duration": generated.audio_seconds, "Output": str(generated.path)})
        except Exception as exc:  # Retain failures in a comparison run instead of losing evidence.
            LOGGER.exception("Benchmark failed model=%s", name)
            rows.append({"Language": language, "Model": name, "Error": str(exc)})
    frame = pd.DataFrame(rows)
    SETTINGS.create_directories()
    csv_path = SETTINGS.benchmark_dir / "benchmark.csv"
    frame.to_csv(csv_path, index=False)
    markdown = "# Benchmark report\n\n" + frame.to_markdown(index=False) + "\n"
    (SETTINGS.benchmark_dir / "benchmark.md").write_text(markdown, encoding="utf-8")
    return frame


def main() -> None:
    """Run a requested benchmark task from the terminal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Reference speaker WAV path")
    parser.add_argument("--language", choices=["en", "hi", "ar"], default="en")
    parser.add_argument("--text", default="This is a reproducible multilingual voice cloning benchmark.")
    parser.add_argument("--models", nargs="+", default=["xtts", "chatterbox", "fishspeech", "mms"])
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    print(benchmark_tts(args.reference, args.models, args.text, args.language, not args.skip_evaluation).to_string(index=False))


if __name__ == "__main__":
    main()
