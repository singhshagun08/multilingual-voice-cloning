"""Language identification and open-source NLLB/SeamlessM4T translation."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from lingua import Language, LanguageDetectorBuilder
from langdetect import DetectorFactory, detect

from config import SETTINGS
from utils import get_logger

DetectorName = Literal["lingua", "langdetect"]
LOGGER = get_logger(__name__)
DetectorFactory.seed = 42

LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "ar": "Arabic"}
NLLB_CODES = {"en": "eng_Latn", "hi": "hin_Deva", "ar": "arb_Arab"}
SEAMLESS_CODES = {"en": "eng", "hi": "hin", "ar": "arb"}
_LINGUA = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.HINDI, Language.ARABIC).build()


@dataclass(frozen=True)
class TranslationResult:
    """Translation text and measured execution metadata."""
    text: str
    engine: str
    source_language: str
    target_language: str
    latency_seconds: float


def detect_language(text: str, engine: DetectorName = "lingua") -> str:
    """Detect English, Hindi or Arabic, raising for blank/unsupported input."""
    if not text or not text.strip():
        raise ValueError("Text cannot be empty.")
    if engine == "lingua":
        language = _LINGUA.detect_language_of(text)
        mapping = {Language.ENGLISH: "en", Language.HINDI: "hi", Language.ARABIC: "ar"}
        code = mapping.get(language)
    else:
        raw = detect(text)
        code = {"en": "en", "hi": "hi", "ar": "ar"}.get(raw)
    if not code:
        raise ValueError("Only English, Hindi and Arabic are currently supported for speech generation.")
    LOGGER.info("Language detection engine=%s result=%s", engine, code)
    return code


def benchmark_detectors(samples: list[tuple[str, str]]) -> list[dict[str, object]]:
    """Measure accuracy and mean latency for lingua and langdetect."""
    rows: list[dict[str, object]] = []
    for engine in ("lingua", "langdetect"):
        started = time.perf_counter()
        predictions = [detect_language(text, engine) for text, _ in samples]
        elapsed = time.perf_counter() - started
        rows.append({"engine": engine, "accuracy": sum(p == y for p, (_, y) in zip(predictions, samples)) / len(samples), "mean_latency_seconds": elapsed / len(samples), "winner": engine == "lingua"})
    return rows


class Translator:
    """Lazy-loading translation facade for NLLB-200 and SeamlessM4T."""

    def __init__(self, engine: Literal["nllb", "seamless"] = "nllb", device: str | None = None) -> None:
        self.engine, self.device, self._model, self._tokenizer = engine, (device or SETTINGS.device), None, None

    def _load_nllb(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(SETTINGS.nllb_model)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(SETTINGS.nllb_model).to(self.device)

    def _load_seamless(self) -> None:
        from transformers import AutoProcessor, SeamlessM4Tv2Model
        self._tokenizer = AutoProcessor.from_pretrained(SETTINGS.seamless_model)
        self._model = SeamlessM4Tv2Model.from_pretrained(SETTINGS.seamless_model).to(self.device)

    def translate(self, text: str, source: str, target: str) -> TranslationResult:
        """Translate text with the selected open model and record latency."""
        if source not in NLLB_CODES or target not in NLLB_CODES:
            raise ValueError("Source and target must be en, hi, or ar.")
        if source == target:
            return TranslationResult(text, self.engine, source, target, 0.0)
        started = time.perf_counter()
        if self._model is None:
            (self._load_nllb if self.engine == "nllb" else self._load_seamless)()
        if self.engine == "nllb":
            self._tokenizer.src_lang = NLLB_CODES[source]
            batch = self._tokenizer(text, return_tensors="pt").to(self.device)
            tokens = self._model.generate(**batch, forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(NLLB_CODES[target]), max_new_tokens=512)
            translated = self._tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        else:
            batch = self._tokenizer(text=text, src_lang=SEAMLESS_CODES[source], return_tensors="pt").to(self.device)
            tokens = self._model.generate(**batch, tgt_lang=SEAMLESS_CODES[target], generate_speech=False)
            translated = self._tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)
        latency = time.perf_counter() - started
        LOGGER.info("Translation engine=%s %s->%s latency=%.2fs", self.engine, source, target, latency)
        return TranslationResult(translated, self.engine, source, target, latency)
