import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

import os
os.environ["MVC_DEVICE"] = "cpu"

from models.xtts_model import XTTSModel
from utils import output_path
from evaluation import ASRTranscriber, normalise_transcript
from jiwer import wer, cer
import torchaudio
import pandas as pd

def load_wav_16k(path):
    signal, sr = torchaudio.load(path)
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
    return signal.mean(dim=0)

xtts = XTTSModel(device="cpu")

text = "مرحباً. تم إنشاء هذا الصوت باستخدام نموذج مفتوح المصدر متعدد اللغات."
language = "ar"

result = xtts.generate(text, language, ["samples/speaker.wav"], output_path(language, "xtts"))
print(f"Generated: {result.path} | RTF={result.rtf:.3f} | gen_time={result.generation_seconds:.1f}s")

transcriber = ASRTranscriber("faster-whisper", device="cpu")
transcript, asr_seconds = transcriber.transcribe(result.path, language)
expected, actual = normalise_transcript(text), normalise_transcript(transcript)
wer_score = float(wer(expected, actual))
cer_score = float(cer(expected, actual))

from speechbrain.inference.speaker import EncoderClassifier
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"})
ref_signal = load_wav_16k("samples/speaker.wav")
gen_signal = load_wav_16k(str(result.path))
emb1 = classifier.encode_batch(ref_signal.unsqueeze(0)).squeeze()
emb2 = classifier.encode_batch(gen_signal.unsqueeze(0)).squeeze()
similarity = torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

print(f"WER={wer_score:.3f} CER={cer_score:.3f} Similarity={similarity:.3f}")
print(f"Transcript: {transcript}")

row = {
    "Language": language, "Model": "xtts",
    "Latency": result.generation_seconds, "RTF": result.rtf,
    "WER": wer_score, "CER": cer_score,
    "Similarity": similarity,
    "Audio Duration": result.audio_seconds,
    "Transcript": transcript,
    "Output": str(result.path)
}
pd.DataFrame([row]).to_csv("benchmark_arabic.csv", index=False)
print("Saved benchmark_arabic.csv")
