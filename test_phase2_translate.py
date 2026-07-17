from translate import Translator

translator = Translator("nllb")

pairs = [
    ("Hello, how are you?", "en", "hi"),
    ("مرحبا، كيف حالك؟", "ar", "en"),
    ("नमस्ते, आप कैसे हैं?", "hi", "ar"),
]

for text, src, tgt in pairs:
    result = translator.translate(text, src, tgt)
    print(f"[{src}->{tgt}] '{text}' => '{result.text}' (latency={result.latency_seconds:.2f}s)")
