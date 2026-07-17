"""Streamlit frontend for open-source multilingual voice cloning and translation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from benchmark import benchmark_tts
from config import SETTINGS
from evaluation import evaluate_audio
from models import MODEL_CLASSES
from router import LanguageRouter
from translate import LANGUAGE_NAMES, Translator, detect_language
from utils import get_logger, output_path

LOGGER = get_logger(__name__)
st.set_page_config(page_title="Open Voice Clone Translate", page_icon="🎙️", layout="wide")


@st.cache_resource(show_spinner=False)
def get_model(name: str):
    """Cache a loaded model across Streamlit reruns."""
    return MODEL_CLASSES[name]()


def save_uploads(files) -> list[Path]:
    """Persist uploaded reference clips to a temporary job directory."""
    root = Path(tempfile.mkdtemp(prefix="mvc_refs_"))
    paths = []
    for file in files:
        path = root / file.name
        path.write_bytes(file.getbuffer())
        paths.append(path)
    return paths


def main() -> None:
    """Render the end-to-end application."""
    SETTINGS.create_directories()
    st.title("🎙️ Multilingual Voice Cloning + Translation")
    st.caption("Open-source NLLB/SeamlessM4T translation and XTTS, Chatterbox, Fish Speech, MMS TTS.")
    left, right = st.columns([3, 2])
    with left:
        references = st.file_uploader("Upload one or more voice reference clips", type=["wav", "mp3", "m4a", "flac"], accept_multiple_files=True)
        text = st.text_area("Text in English, Hindi, or Arabic", height=150, placeholder="Type or paste text…")
    with right:
        target = st.selectbox("Output language", options=list(LANGUAGE_NAMES), format_func=LANGUAGE_NAMES.get)
        translation_engine = st.selectbox("Translation model", ["nllb", "seamless"])
        model_choice = st.selectbox("TTS model", ["auto", *MODEL_CLASSES])
        asr_engine = st.selectbox("Evaluation ASR", ["faster-whisper", "whisper", "indicwhisper"])
        evaluate = st.checkbox("Calculate WER/CER and speaker similarity", value=True)
    if st.button("Generate cloned speech", type="primary", use_container_width=True):
        if not text.strip():
            st.error("Enter text before generating speech.")
            return
        if not references:
            st.error("Upload at least one reference clip for voice cloning.")
            return
        try:
            progress = st.progress(0, text="Detecting language…")
            source = detect_language(text)
            st.info(f"Detected language: **{LANGUAGE_NAMES[source]}**")
            progress.progress(15, text="Translating text…")
            translated = Translator(translation_engine).translate(text, source, target)
            progress.progress(45, text="Preparing cloned voice model…")
            chosen = LanguageRouter.default().choose(target, None if model_choice == "auto" else model_choice)
            refs = save_uploads(references)
            model = get_model(chosen)
            result = model.generate(translated.text, target, refs, output_path(LANGUAGE_NAMES[target], chosen))
            progress.progress(80, text="Evaluating generated speech…")
            metrics = evaluate_audio(result.path, translated.text, target, refs[0], asr_engine) if evaluate else None
            progress.progress(100, text="Complete")
            st.audio(str(result.path))
            st.download_button("Download WAV", data=result.path.read_bytes(), file_name=result.path.name, mime="audio/wav")
            values = {"Model": result.model, "Translation latency (s)": round(translated.latency_seconds, 2), "Generation time (s)": round(result.generation_seconds, 2), "Audio duration (s)": round(result.audio_seconds, 2), "RTF": round(result.rtf, 3)}
            if metrics:
                values.update({"WER": round(metrics.wer, 3), "CER": round(metrics.cer, 3), "Speaker similarity": round(metrics.speaker_similarity or 0.0, 3), "ASR transcript": metrics.transcript, "MOS": metrics.mos})
            st.dataframe(pd.DataFrame([values]), use_container_width=True)
            LOGGER.info("Generated path=%s model=%s", result.path, chosen)
        except Exception as exc:
            LOGGER.exception("Generation failed")
            st.exception(exc)
    st.divider()
    st.subheader("Saved benchmark results")
    benchmark_path = SETTINGS.benchmark_dir / "benchmark.csv"
    if benchmark_path.exists():
        st.dataframe(pd.read_csv(benchmark_path), use_container_width=True)
    else:
        st.caption("Run `python benchmark.py --reference samples/speaker.wav` to create a benchmark table.")
    with st.expander("Application logs"):
        log_path = SETTINGS.log_dir / "application.log"
        st.code(log_path.read_text(encoding="utf-8")[-12000:] if log_path.exists() else "No log entries yet.")


if __name__ == "__main__":
    main()
