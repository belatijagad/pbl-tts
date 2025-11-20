import csv
import json
import random
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from pydub import AudioSegment

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


def download_and_extract_tgz_dataset(
    repo_id: str,
    filename: str,
    download_dir: str,
) -> Path:
    """Download a .tgz dataset artifact from Hugging Face and extract it in-place."""

    data_dir = Path(download_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=data_dir,
        )
    )

    extract_dir = artifact_path.parent
    with tarfile.open(artifact_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    artifact_path.unlink()
    return extract_dir

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


def convert_language_subset_to_wav(
    dataset_root: str,
    language: str = "indonesian",
    split_subdir: str = "test",
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    sample_rate: int = 22050,
) -> List[Path]:
    """Convert all MP3 files for a given language subset into WAV format."""

    dataset_root_path = Path(dataset_root)
    mp3_dir = dataset_root_path / split_subdir / language
    if not mp3_dir.exists():
        raise FileNotFoundError(f"MP3 directory not found: {mp3_dir}")

    if output_dir is None:
        output_dir_path = dataset_root_path / f"processed_{language}" / "wavs"
    else:
        output_dir_path = Path(output_dir)

    converted_paths: List[Path] = []
    for mp3_file in mp3_dir.rglob("*.mp3"):
        relative_path = mp3_file.relative_to(mp3_dir).with_suffix(".wav")
        wav_path = output_dir_path / relative_path
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        if wav_path.exists() and not overwrite:
            converted_paths.append(wav_path)
            continue

        audio = AudioSegment.from_mp3(str(mp3_file))
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        audio.export(str(wav_path), format="wav")
        converted_paths.append(wav_path)

    return converted_paths


def load_librivox_metadata(metadata_csv_path: str, language: str = "indonesian") -> List[Dict]:
    """Load metadata entries filtered by language from the Librivox manifest."""

    entries: list[dict] = []
    with open(metadata_csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 4:
                continue
            audio_rel_path, lang_code, speaker_id, text = row[:4]
            if f"/{language}/" not in audio_rel_path:
                continue
            entries.append(
                {
                    "audio_rel_path": audio_rel_path,
                    "language": lang_code,
                    "speaker_id": speaker_id,
                    "text": text.strip(),
                }
            )
    return entries


def create_manifest_from_librivox(
    metadata_csv_path: str,
    wav_root: str,
    manifest_path: str,
    language: str = "indonesian",
    mel_dir: Optional[str] = None,
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
) -> list[dict]:
    """Create a NeMo style manifest from Librivox metadata and converted WAVs."""

    wav_root_path = Path(wav_root)
    if mel_dir is not None:
        mel_root_path = Path(mel_dir)
        mel_root_path.mkdir(parents=True, exist_ok=True)
    else:
        mel_root_path = None

    metadata_entries = load_librivox_metadata(metadata_csv_path, language=language)
    manifest_entries: List[Dict] = []

    for entry in metadata_entries:
        mp3_rel_path = Path(entry["audio_rel_path"])
        try:
            relative_to_language = Path(*mp3_rel_path.parts[mp3_rel_path.parts.index(language) + 1 :])
        except ValueError as exc:
            raise ValueError(f"Language '{language}' not found in path {mp3_rel_path}") from exc

        wav_path = wav_root_path / relative_to_language.with_suffix(".wav")
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing WAV file for {wav_path}")

        audio_info = sf.info(str(wav_path))
        manifest_record = {
            "audio_filepath": str(wav_path.absolute()),
            "duration": audio_info.duration,
            "text": entry["text"],
            "normalized_text": entry["text"],
            "speaker": entry["speaker_id"],
            "language": entry["language"],
        }

        if mel_root_path is not None:
            relative_wav = wav_path.relative_to(wav_root_path)
            mel_path = mel_root_path / relative_wav.with_suffix(".pt")
            mel_path.parent.mkdir(parents=True, exist_ok=True)
            mel_tensor = generate_mel_spectrogram(
                str(wav_path),
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
            )
            torch.save(mel_tensor, mel_path)
            manifest_record["mel_filepath"] = str(mel_path.absolute())
            manifest_record["mel_path"] = str(mel_path.absolute())

        manifest_entries.append(manifest_record)

    with open(manifest_path, "w", encoding="utf-8") as fout:
        for record in manifest_entries:
            fout.write(json.dumps(record) + "\n")

    return manifest_entries


def split_manifest_entries(
    entries: List[Dict],
    train_manifest_path: str,
    eval_manifest_path: str,
    train_size: int = 140,
    eval_size: int = 10,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Split manifest entries into fixed train/eval subsets and write them out."""

    if train_size + eval_size > len(entries):
        raise ValueError(
            f"Requested split larger than available entries: {train_size + eval_size} > {len(entries)}"
        )

    rng = random.Random(seed)
    shuffled_entries = entries.copy()
    rng.shuffle(shuffled_entries)

    train_entries = shuffled_entries[:train_size]
    eval_entries = shuffled_entries[train_size : train_size + eval_size]

    with open(train_manifest_path, "w", encoding="utf-8") as train_out:
        for record in train_entries:
            train_out.write(json.dumps(record) + "\n")

    with open(eval_manifest_path, "w", encoding="utf-8") as eval_out:
        for record in eval_entries:
            eval_out.write(json.dumps(record) + "\n")

    return train_entries, eval_entries
