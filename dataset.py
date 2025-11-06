import json
import random
import shutil
import zipfile
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download

def fetch_dataset(repo_id: str,
                  filename: str="sundanese/su_id_male_01596.zip",
                  repo_type: str="dataset",
                  download_dir: str="./data") -> None:
    
    data_dir = Path(download_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    file_path_str = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
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


def create_manifest(data_dir: str, transcript_path: str, manifest_path: str):
    
    data_path = Path(data_dir)
    audio_dir = data_path / "wavs"
    
    transcripts = {}
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            filename, text = line.strip().split("\t", 1)
            transcripts[filename] = text.lower()

    with open(manifest_path, "w", encoding="utf-8") as fout:
        for audio_file in audio_dir.glob("*.wav"): 
            info = sf.info(str(audio_file))
            filename = audio_file.name
            
            entry = {
                "audio_filepath": str(audio_file.absolute()),
                "duration": info.duration,
                "text": transcripts[filename] 
            }
            fout.write(json.dumps(entry) + "\n")
