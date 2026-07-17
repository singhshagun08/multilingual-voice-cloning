import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])

from models.xtts_model import XTTSModel
from utils import output_path

xtts = XTTSModel(device="cpu")

examples = {
    "en": "Hello. This voice was generated with an open source multilingual model.",
    "hi": "नमस्ते। यह आवाज़ एक ओपन सोर्स बहुभाषी मॉडल से बनाई गई है।",
    "ar": "مرحباً. تم إنشاء هذا الصوت باستخدام نموذج مفتوح المصدر متعدد اللغات.",
}

for language, sentence in examples.items():
    result = xtts.generate(sentence, language, ["samples/speaker.wav"], output_path(language, "xtts"))
    print(f"{language}: {result.path} | RTF={result.rtf:.3f} | gen_time={result.generation_seconds:.1f}s")
