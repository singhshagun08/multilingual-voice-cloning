import glob
import torch
import torchaudio
from evaluation import SpeakerSimilarity

files = sorted(glob.glob("outputs/en_xtts_*.wav"))
if not files:
    raise FileNotFoundError("Run test_phase2_xtts.py first.")
generated = files[-1]

def load_wav(path):
    signal, sr = torchaudio.load(path)
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
    return signal.mean(dim=0)  # mono

scorer = SpeakerSimilarity(device="cpu")
scorer._load()  # force-load the classifier without going through load_audio

ref_signal = load_wav("samples/speaker.wav")
gen_signal = load_wav(generated)

first = scorer.classifier.encode_batch(ref_signal.unsqueeze(0)).squeeze()
second = scorer.classifier.encode_batch(gen_signal.unsqueeze(0)).squeeze()
score = torch.nn.functional.cosine_similarity(first.unsqueeze(0), second.unsqueeze(0)).item()

print(f"Speaker similarity (cosine): {score:.3f}")
