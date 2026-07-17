
# NOTE: XTTS and Chatterbox require incompatible transformers versions
# (XTTS needs ~4.5x, Chatterbox needs exactly 5.2.0). Importing this package
# with both installed simultaneously will fail. See README section 7.5/7.2.

"""TTS wrappers exposed to the language router."""
from .xtts_model import XTTSModel
from .chatterbox_model import ChatterboxModel
from .fishspeech_model import FishSpeechModel
from .mms_model import MMSModel

MODEL_CLASSES = {"xtts": XTTSModel, "chatterbox": ChatterboxModel, "fishspeech": FishSpeechModel, "mms": MMSModel}
