from pathlib import Path
import torch
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from nemo.collections.tts.models.vits import VitsModel
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="nemo")
warnings.filterwarnings("ignore", module="pytorch_lightning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory to save .nemo models
save_dir = Path("./nemo_pretrained")
save_dir.mkdir(exist_ok=True)

# Fetch and Load Models
fastpitch = FastPitchModel.from_pretrained("tts_en_fastpitch")
hifigan = HifiGanModel.from_pretrained("tts_en_hifigan")
vits = VitsModel.from_pretrained("tts_en_lj_vits")

# Save .nemo files and store their paths
fastpitch_path = save_dir / "tts_en_fastpitch.nemo"
hifigan_path = save_dir / "tts_en_hifigan.nemo"
vits_path = save_dir / "tts_en_lj_vits.nemo"

print(f"Saving models to {save_dir} ...")

fastpitch.save_to(fastpitch_path)
hifigan.save_to(hifigan_path)
vits.save_to(vits_path)