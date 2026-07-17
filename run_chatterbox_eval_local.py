import os
os.environ["MVC_DEVICE"] = "cpu"

from evaluation import ASRTranscriber, SpeakerSimilarity, normalise_transcript
from jiwer import wer, cer
import pandas as pd

audio_path = "outputs/en_chatterbox_40489b34.wav"
text = "Hello. This voice was generated with an open source multilingual model."

transcriber = ASRTranscriber("faster-whisper", device="cpu")
transcript, asr_seconds = transcriber.transcribe(audio_path, "en")
expected, actual = normalise_transcript(text), normalise_transcript(transcript)
wer_score = float(wer(expected, actual))
cer_score = float(cer(expected, actual))

similarity = SpeakerSimilarity(device="cpu").score("samples/speaker.wav", audio_path)

print(f"WER={wer_score:.3f} CER={cer_score:.3f} Similarity={similarity:.3f}")
print(f"Transcript: {transcript}")

row = {
    "Language": "en", "Model": "chatterbox",
    "Latency": 157.2, "RTF": 35.716,
    "WER": wer_score, "CER": cer_score,
    "Similarity": similarity,
    "Audio Duration": 4.402,
    "Transcript": transcript,
    "Output": audio_path
}
pd.DataFrame([row]).to_csv("benchmark_chatterbox_cpu.csv", index=False)
print("Saved benchmark_chatterbox_cpu.csv")
