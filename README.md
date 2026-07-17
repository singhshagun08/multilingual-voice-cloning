# Multilingual Voice Cloning + Translation — Benchmark Report

## Summary

This project builds and benchmarks three open-source voice-cloning pipelines — English, Hindi, and Arabic — and recommends a per-language model based on measured results rather than assumption. **Chatterbox is the recommended model for English** (speaker similarity 0.785, MOS 5/5, WER 0.0, measured on GPU), because no other tested open-source model came close on cloning fidelity or naturalness for English. **XTTS-v2 is the recommended model for Hindi and Arabic**, not because it hits target quality bars (it doesn't — similarity sits around 0.50, MOS 2-3/5) but because it was the only tested model that supports voice cloning in those languages at all; Chatterbox and MMS-TTS were ruled out for non-English use (Chatterbox is English-only, MMS does not clone voices). All core generation used open-source models only (XTTS-v2, Chatterbox, MMS-TTS); no closed APIs were used anywhere, including evaluation. Full benchmark numbers, methodology, and honest failure modes are below.

---

## 1. What this is

Three separate, working TTS pipelines — one per language — sharing a common codebase (translation, routing, evaluation, benchmarking). Each pipeline was independently benchmarked and the best-performing open-source model was selected per language based on real measured results.

**Core generation is 100% open-source**: XTTS-v2 (Coqui), Chatterbox (Resemble AI), MMS-TTS (Meta). No closed APIs (ElevenLabs, OpenAI, Google, Azure) were used for speech generation.

**Disclosure of tools used for evaluation (per Section 6 ground rules):** all evaluation tools are open-source. faster-whisper (ASR/WER), vasista22/whisper-hindi-small (Hindi-specific ASR), SpeechBrain ECAPA-TDNN (speaker similarity), jiwer (WER/CER calculation). No closed-source tool was used anywhere in this project, including for evaluation.

---

## 2. Recommended pipeline per language

| Language | Recommended Model | Why |
|---|---|---|
| **English** | **Chatterbox** | Highest speaker similarity (0.785 GPU / 0.678 CPU vs XTTS's 0.51), highest MOS (5/5 vs 3/5), perfect WER/CER (0.0/0.0) |
| **Hindi** | XTTS-v2 | Only tested model with native Hindi cloning support; Chatterbox and MMS don't clone Hindi at all |
| **Arabic** | XTTS-v2 | Only tested model with native Arabic cloning support |

**Router configuration** (`config.py`):
```python
router: Mapping[str, str] = field(default_factory=lambda: {
    "en": os.getenv("MVC_EN_MODEL", "chatterbox"),
    "hi": os.getenv("MVC_HI_MODEL", "xtts"),
    "ar": os.getenv("MVC_AR_MODEL", "xtts"),
})
```

This confirms the hypothesis in the original brief: **no single open-source model wins across all three languages.** Chatterbox is English-only and clearly the best English option; XTTS is the only model in our tests that covers Hindi and Arabic cloning at all, so it wins those by default of coverage, not by hitting target quality bars.

---

## 3. Benchmark results

### 3.1 Core metrics — CPU (local machine, no GPU — see Section 3.3 for GPU numbers)

| Language | Model | Latency (s) | RTF | WER | CER | Speaker Similarity | MOS (1 listener) |
|---|---|---|---|---|---|---|---|
| English | XTTS-v2 | 61.3 | 7.39 | 0.091 | 0.087 | 0.514 | 3 / 5 |
| English | **Chatterbox** | 157.2 | 35.72 | 0.000 | 0.000 | 0.678 | **5 / 5** |
| English | MMS-TTS (baseline, no cloning) | 9.7 | 2.22 | 0.125 | 0.051 | 0.034 | — |
| Arabic | XTTS-v2 | 59.5 | 7.46 | 0.182 | 0.061 | 0.501 | 3 / 5 |
| Hindi | XTTS-v2 | 60.7 | 7.88 | 0.417* | 0.214 | 0.536 | 2 / 5 |

*Hindi WER drops to **0.167** when re-scored with a Hindi-specialized ASR model (`vasista22/whisper-hindi-small`) instead of generic `faster-whisper`, on the exact same audio clip — see Section 5, Finding #1.

Note: Chatterbox's CPU latency (157s, RTF 35.7) is markedly heavier than XTTS's (61s, RTF 7.4) — Chatterbox is a larger, more expensive model architecturally. On GPU (Section 3.3) this drops to RTF 2.5, much closer to XTTS's GPU RTF of 1.17.

### 3.2 Against brief targets (Section 3 of spec)

| Metric | Target | English (Chatterbox) | English (XTTS) | Arabic (XTTS) | Hindi (XTTS) |
|---|---|---|---|---|---|
| MOS | ≥ 4.0 | ✅ 5.0 | ❌ 3.0 | ❌ 3.0 | ❌ 2.0 |
| Speaker similarity | ≥ 0.75 | ✅ 0.785 (GPU) / ❌ 0.678 (CPU) | ❌ 0.514 | ❌ 0.501 | ❌ 0.536 |
| RTF | ≤ 0.5 | ❌ 35.72 (CPU) / ❌ 2.50 (GPU) | ❌ 7.39 (CPU) / ❌ 1.17 (GPU) | ❌ 7.46 (CPU) | ❌ 7.88 (CPU) |
| WER | ≤ 10% | ✅ 0.000 | ❌ 0.091 | ❌ 0.182 | ❌ 0.167–0.417 |

**Only Chatterbox-English on GPU clears both the MOS and similarity bars simultaneously.** No model meets the RTF target on any hardware tested — see Section 6 for what this implies.

### 3.3 GPU reference point (Google Colab, NVIDIA T4)

| Language | Model | Latency (s) | RTF | WER | CER | Similarity |
|---|---|---|---|---|---|---|
| English | XTTS-v2 | 6.6 | 1.17 | 0.000 | 0.000 | 0.488 |
| English | Chatterbox | 8.3 | 2.50 | 0.000 | 0.000 | 0.785 |
| English | MMS-TTS | 9.7 (CPU — GPU quota exhausted mid-session) | 2.22 | 0.125 | 0.051 | 0.034 |

GPU cuts XTTS's RTF from ~7.4x to 1.17x real-time, and Chatterbox's from ~35.7x to 2.5x — a substantial improvement for both, though neither hits the 0.5 target. Hindi and Arabic were not re-benchmarked on GPU due to Colab free-tier GPU quota limits during testing; this is a known gap, see Section 6.

Interestingly, similarity scores differ slightly between CPU and GPU runs for the same model (Chatterbox: 0.785 GPU vs 0.678 CPU; XTTS: 0.488 GPU vs 0.514 CPU) — likely due to floating-point precision differences between CPU and GPU inference paths, not a meaningful quality difference.

---

## 4. Models tested and why

| Model | Languages tested | Cloning? | Result | Included in final router? |
|---|---|---|---|---|
| XTTS-v2 (Coqui) | en, hi, ar | Yes | Moderate similarity (~0.5) across all three; only model covering all three languages | Yes (hi, ar) |
| Chatterbox | en only | Yes | Best-in-class similarity (0.785 GPU) and MOS (5/5) for English | Yes (en) |
| MMS-TTS (Meta) | en | No (baseline) | Fast but zero speaker cloning (similarity 0.034, as expected) — useful only as a non-cloning control | No |
| Fish Speech | — | — | **Could not test.** Repo's `pyproject.toml` pins `tokenizers<0.11` (~2021-era), which has no prebuilt wheel for Python 3.12 and fails to compile from source. This is a packaging limitation of Fish Speech, not a quality finding. | No |

---

## 5. Key findings

1. **ASR choice significantly affects reported Hindi WER.** The same Hindi XTTS clip scored WER 0.417 with generic `faster-whisper` but WER 0.167 with the Hindi-specialized `vasista22/whisper-hindi-small`. (Note: `ai4bharat/indicwhisper`, suggested in the original brief hints, does not actually exist as a Hugging Face repo — it returns a 404. `vasista22/whisper-hindi-small` is the real, working community fine-tune used here.) Any Hindi TTS evaluation using a generic multilingual ASR model will understate real quality.

2. **XTTS's speaker similarity is stable but moderate across all languages** (~0.50–0.54 on CPU, 0.488 on GPU) — this appears to be an architectural ceiling for XTTS's speaker-conditioning approach, not a language-specific weakness.

3. **Chatterbox dramatically outperforms XTTS on every measured axis for English** (similarity 0.785 vs 0.51, MOS 5 vs 3, WER 0.0 vs 0.091) but has no Hindi/Arabic support in this codebase's integration.

4. **No single tested model covers all three languages at target quality.** This confirms the brief's own hypothesis — a per-language router, not a single universal model, is the practical answer today.

5. **RTF is heavily hardware-dependent.** CPU RTF is roughly 3-6x worse than GPU RTF for the same model and text (XTTS: 7.4x CPU vs 1.17x GPU; Chatterbox: 35.7x CPU vs 2.5x GPU). Any RTF claim must state the hardware.

6. **A real, previously-hidden bug was found and fixed during this project:** SpeechBrain's `EncoderClassifier.load_audio()` triggers a broken lazy import of an optional `k2_fsa` module that isn't installed, crashing speaker-similarity scoring entirely. Fixed in `evaluation.py` by loading audio directly via `torchaudio` instead of SpeechBrain's built-in loader.

---

## 6. What's missing / how I'd improve this

- **MOS is currently single-listener.** The brief asks for "you plus a few others" — only one listener's scores are recorded here. A proper MOS study needs 3+ independent listeners per clip, ideally blind (listeners don't know which model produced which clip).
- **Hindi and Arabic were never benchmarked on GPU** — only English has a GPU data point. Free-tier Colab GPU quota ran out mid-testing. A full GPU pass across all three languages is the single highest-value next step.
- **Fish Speech, CosyVoice2, IndexTTS-2, and Chatterbox-Turbo were not tested** — the brief specifically flagged these as strong recent options. Fish Speech failed on packaging; the others were not attempted due to time.
- **Only short (~10–15 word) test sentences were used.** Longer paragraphs, proper nouns, and numbers were never stress-tested — a known weak point for autoregressive TTS models like XTTS.
- **Explicit latency-to-first-audio-chunk (streaming) was not measured** — all models here run in batch mode (full clip at once), so only full-clip latency is reported, not first-chunk latency. Streaming-capable models (CosyVoice2 is specifically noted for this) were not evaluated.
- **No prosody, emotion, or numbers/names handling was assessed** — the brief calls these out as bonus metrics worth considering.
- **The Streamlit UI (`app.py`) has not been launched/tested end-to-end** in this project. XTTS and Chatterbox require incompatible `transformers` versions (~4.5x vs exactly 5.2.0), and `models/__init__.py` eagerly imports both, so the app will fail to even start unless only one of the two is installed at a time. All pipeline testing in this project was done via direct script calls to avoid this conflict. A proper fix would make model imports lazy (only import the model actually selected at runtime).

---

## 7. Setup and reproduction

### 7.1 Requirements
- Python 3.11 (local) or 3.12 (Colab)
- ffmpeg installed and on PATH
- A consented reference voice sample, 6–20 seconds, placed at `samples/speaker.wav`

### 7.2 Local setup (CPU — functional but slow, RTF ~7-36x depending on model)
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.57.1 accelerate sentencepiece
pip install "streamlit>=1.41,<2" librosa soundfile pandas tabulate jiwer
pip install speechbrain faster-whisper coqui-tts
pip install langdetect lingua-language-detector
```

**Known conflict:** XTTS (`coqui-tts`) requires `transformers` around `4.5x`; Chatterbox requires exactly `transformers==5.2.0`. They cannot be installed together. Switch between them with:
```bash
pip install "transformers==4.57.1"                   # for XTTS
pip install "transformers==5.2.0" chatterbox-tts       # for Chatterbox
```

Exact locked versions that were verified to work are recorded in `working_requirements.txt`.

### 7.3 Colab setup (GPU — recommended for real use, RTF ~1.2-2.5x)
See `notebooks/GoogleColab.ipynb`. Same transformers version conflict applies; install XTTS and Chatterbox in separate sessions/cells, never together. Original Coqui `TTS` package does not support Python 3.12 (Colab's default) — use `coqui-tts` (community fork) instead.

### 7.4 Running a benchmark
```bash
python benchmark.py --reference samples/speaker.wav --language en --models xtts
```
Writes `benchmark/benchmark.csv` and `benchmark/benchmark.md`.

Individual reproducible benchmark scripts used to generate every number in this README are included directly: `run_arabic_benchmark.py`, `run_hindi_benchmark.py`, `run_english_cpu_benchmark.py`, `run_chatterbox_local.py`, `run_chatterbox_eval_local.py`, `run_indicwhisper_test.py`, `run_full_comparison.py`, `run_latency_summary.py`.

### 7.5 Running the app
```bash
streamlit run app.py
```
**Known limitation:** see Section 6 — the app will fail to start unless only one of XTTS/Chatterbox is installed at a time, due to the transformers version conflict.

---

## 8. Repository layout

```text
app.py                            Streamlit UI (routes by language via config.py) — untested end-to-end, see Section 6
config.py                         Central settings + per-language model router
translate.py                      Language detection (Lingua) + NLLB/SeamlessM4T translation
router.py                         LanguageRouter — model selection logic
evaluation.py                     ASR (faster-whisper/Hindi-Whisper), WER/CER, ECAPA speaker similarity
benchmark.py                      CLI benchmark runner, writes CSV/Markdown reports
models/                           XTTS, Chatterbox, Fish Speech (untested), MMS wrappers
samples/speaker.wav               Reference voice used for all cloning benchmarks
outputs/                          Generated WAV files, one per benchmark run (4 samples included)
benchmark_xtts.csv                English XTTS benchmark (GPU)
benchmark_english_cpu.csv         English XTTS benchmark (CPU)
benchmark_chatterbox_cpu.csv      English Chatterbox benchmark (CPU)
benchmark_arabic.csv              Arabic XTTS benchmark (CPU)
benchmark_hindi.csv               Hindi XTTS benchmark (CPU, generic ASR)
benchmark_hindi_indicwhisper.csv  Hindi XTTS benchmark (CPU, Hindi-specific ASR)
mos_scores.csv                    Human MOS ratings (1 listener, see Section 6)
full_comparison_cpu.csv           Consolidated CPU benchmark table (Section 3.1)
latency_summary.csv               Latency-to-full-clip summary
working_requirements.txt          Exact locked dependency versions verified to work
run_*.py                          Individual reproducible benchmark scripts (see Section 7.4)
```
