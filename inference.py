import torch
import librosa
import numpy as np
from nemo.collections.tts.models.vits import VitsModel
from nemo.collections.tts.data.dataset import TTSDataset
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder

def inference(
    batches: TTSDataset, 
    sample_rate: int, 
    score_fn: callable,
    e2e_model: VitsModel | None = None,
    generator: SpectrogramGenerator | None = None, 
    vocoder: Vocoder | None = None, 
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """https://colab.research.google.com/github/NVIDIA/NeMo/blob/v1.0.0/tutorials/tts/1_TTS_inference.ipynb
    """
    assert (e2e_model is not None) + (generator is not None and vocoder is not None) == 1, \
        "You must provide either e2e_model or (generator AND vocoder)."
    
    scores, audio_preds = [], []
    with torch.inference_mode():
        for b in batches:
            parser = generator or e2e_model
            tokens = parser.parse(b["text"])

            audio_gt, _ = librosa.load(b["audio_path"], sr=sample_rate)
            audio_gt_tensor = torch.from_numpy(audio_gt).unsqueeze(0).to(parser.device)
            audio_gt_len = torch.tensor(audio_gt_tensor.shape[1], dtype=torch.long, device=parser.device).unsqueeze(0)
            
            if e2e_model is None:
                spectogram = generator.generate_spectrogram(tokens)
                real_spectogram = generator.preprocessor(input_signal=audio_gt_tensor, length=audio_gt_len)
                audio_pred = vocoder.convert_spectrogram_to_audio(spectogram)
            else:
                audio_pred = e2e_model.convert_text_to_waveform(tokens)
                audio_pred_len = torch.tensor(audio_pred.shape[1], dtype=torch.long, device=parser.device).unsqueeze(0)
                spectogram = e2e_model.audio_to_melspec_processor(audio_pred, audio_pred_len)
                real_spectogram = e2e_model.audio_to_melspec_processor(audio_gt_tensor, audio_gt_len)

            scores.append(score_fn(real_spectogram, spectogram))
            audio_preds.append(audio_pred.squeeze().cpu().numpy())

    return audio_preds, scores
