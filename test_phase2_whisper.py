import glob
from evaluation import ASRTranscriber

files = sorted(glob.glob("outputs/en_xtts_*.wav"))
if not files:
    raise FileNotFoundError("Run test_phase2_xtts.py first to generate an English sample.")
audio_path = files[-1]

transcriber = ASRTranscriber("faster-whisper", device="cpu")
transcript, seconds = transcriber.transcribe(audio_path, "en")
print(f"Transcript: {transcript}")
print(f"ASR time: {seconds:.2f}s")
