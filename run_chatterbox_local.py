import sys, types, importlib.util

fake_models_pkg = types.ModuleType("models")
fake_models_pkg.__path__ = ["models"]
sys.modules["models"] = fake_models_pkg

spec_base = importlib.util.spec_from_file_location("models.base", "models/base.py")
base_module = importlib.util.module_from_spec(spec_base)
sys.modules["models.base"] = base_module
spec_base.loader.exec_module(base_module)

spec_cb = importlib.util.spec_from_file_location("models.chatterbox_model", "models/chatterbox_model.py")
chatterbox_module = importlib.util.module_from_spec(spec_cb)
sys.modules["models.chatterbox_model"] = chatterbox_module
spec_cb.loader.exec_module(chatterbox_module)

ChatterboxModel = chatterbox_module.ChatterboxModel

import os
os.environ["MVC_DEVICE"] = "cpu"
from utils import output_path

model = ChatterboxModel(device="cpu")
text = "Hello. This voice was generated with an open source multilingual model."
result = model.generate(text, "en", ["samples/speaker.wav"], output_path("en", "chatterbox"))
print(f"Generated: {result.path} | RTF={result.rtf:.3f} | gen_time={result.generation_seconds:.1f}s")
