import json
import random
import shutil
import zipfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

def fetch_dataset(repo_id: str, filename: str, download_dir: str) -> None:
    data_dir = Path(download_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path_str = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        local_dir=data_dir,
    )
    zip_path = Path(file_path_str)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    redundant_name = zip_path.stem 
    redundant_dir = data_dir / redundant_name

    if redundant_dir.is_dir():
        for item in redundant_dir.iterdir():
            shutil.move(str(item), str(data_dir / item.name))
        redundant_dir.rmdir()

    zip_path.unlink() 
    zip_path.parent.rmdir()

def split_data(full_manifest_path: str, 
               train_manifest_path: str, 
               eval_manifest_path: str, 
               eval_size: int = 10, 
               seed: int = 42) -> None:    
    with open(full_manifest_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()

    random.seed(seed)
    random.shuffle(all_lines)

    if eval_size > len(all_lines):
        print(f"Warning: eval_size ({eval_size}) is larger than total data ({len(all_lines)}).")
        eval_lines = all_lines
        train_lines = []
    else:
        eval_lines = all_lines[:eval_size]
        train_lines = all_lines[eval_size:]

    with open(train_manifest_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(eval_manifest_path, "w", encoding="utf-8") as f:
        f.writelines(eval_lines)


def generate_mel_spectrogram(
    audio_path: str,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: int = 0,
    fmax: int = 8000,
    window: str = "hann",
) -> torch.Tensor:
    """Generate mel spectrogram from audio file and return as torch tensor."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Generate mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        window=window,
    )
    
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_tensor = torch.from_numpy(mel_db).float()
    return mel_tensor


def create_manifest(data_dir: str, transcript_path: str, manifest_path: str, mel_dir: str = None):
    
    data_path = Path(data_dir)
    audio_dir = data_path / "wavs"
    
    # Create mel directory if not specified
    if mel_dir is None:
        mel_dir = data_path / "mels"
    else:
        mel_dir = Path(mel_dir)
    mel_dir.mkdir(parents=True, exist_ok=True)
    
    transcripts = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            filename, text = line.strip().split("\t", 1)
            transcripts[filename] = text.lower()

    with open(manifest_path, "w", encoding="utf-8") as fout:
        for audio_file in audio_dir.glob("*.wav"): 
            info = sf.info(str(audio_file))
            filename = audio_file.name.removesuffix('.wav')
            
            # Generate and save mel spectrogram
            mel_tensor = generate_mel_spectrogram(str(audio_file))
            mel_path = mel_dir / f"{filename}.pt"
            torch.save(mel_tensor, mel_path)
            
            entry = {
                "audio_filepath": str(audio_file.absolute()),
                "duration": info.duration,
                "text": transcripts[filename],
                "normalized_text": transcripts[filename],
                "mel_path": str(mel_path.absolute()),
                "mel_filepath": str(mel_path.absolute())
            }
            fout.write(json.dumps(entry) + "\n")
