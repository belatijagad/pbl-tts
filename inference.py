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
            audio_gt, audio_gt_len, text, text_len, *_ = b
            audio_gt = audio_gt.to(parser.device)
            audio_gt_len = audio_gt_len.to(parser.device)
            text = text.to(parser.device)
            text_len = text_len.to(parser.device)

            print("--------------------------------")
            print(audio_gt)
            print(audio_gt_len)
            print(text)
            print(text_len)
            print('--------------------------------')
            
            # text ~ (seq_len,)
            if e2e_model is None:
                spectogram = generator.generate_spectrogram(text.unsqueeze(0))
                gt_mels, gt_mel_len = generator.preprocessor(input_signal=audio_gt.unsqueeze(0), length=audio_gt_len.unsqueeze(0))
                audio_pred = vocoder.convert_spectrogram_to_audio(spec=spectogram)
            else:
                audio_pred = e2e_model.convert_text_to_waveform(text)
                audio_pred_len = torch.tensor(audio_pred.shape[1], dtype=torch.long, device=parser.device).unsqueeze(0)
                spectogram = e2e_model.audio_to_melspec_processor(audio_pred, audio_pred_len)
                gt_mels, gt_mel_len = e2e_model.audio_to_melspec_processor(audio_gt, audio_gt_len)

            print(gt_mels)
            scores.append(score_fn(gt_mels.squeeze().cpu().numpy(), spectogram.squeeze().cpu().numpy()))
            audio_preds.append(audio_pred.squeeze().cpu().numpy())

    return audio_preds, scores
