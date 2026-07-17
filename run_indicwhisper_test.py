import os, re
os.environ["MVC_DEVICE"] = "cpu"

from transformers import pipeline
from jiwer import wer, cer
import glob

def normalise_transcript(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s\u0600-\u06ff]", "", text.lower())).strip()

audio_files = sorted(glob.glob("outputs/hi_xtts_*.wav"))
if not audio_files:
    raise FileNotFoundError("Run run_hindi_benchmark.py first to generate a Hindi clip.")
audio_path = audio_files[-1]
print(f"Using audio: {audio_path}")

text = "नमस्ते। यह आवाज़ एक ओपन सोर्स बहुभाषी मॉडल से बनाई गई है।"

print("Loading Hindi-tuned Whisper-small model...")
asr = pipeline("automatic-speech-recognition", model="vasista22/whisper-hindi-small", chunk_length_s=30, device=-1)

result = asr(audio_path)
transcript = result["text"].strip()

expected, actual = normalise_transcript(text), normalise_transcript(transcript)
wer_score = float(wer(expected, actual))
cer_score = float(cer(expected, actual))

print(f"Hindi-Whisper-small transcript: {transcript}")
print(f"WER={wer_score:.3f} CER={cer_score:.3f}")

import pandas as pd
row = {"Language": "hi", "Model": "xtts", "ASR": "vasista22/whisper-hindi-small", "WER": wer_score, "CER": cer_score, "Transcript": transcript}
pd.DataFrame([row]).to_csv("benchmark_hindi_indicwhisper.csv", index=False)
print("Saved benchmark_hindi_indicwhisper.csv")
