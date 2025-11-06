import shutil
from pathlib import Path

import torch
import hydra
import soundfile as sf
from omegaconf import DictConfig, OmegaConf
from nemo.collections.tts.models.vits import VitsModel
from nemo.collections.tts.data.dataset import TTSDataset
from nemo.collections.tts.models import FastPitchModel, HifiGanModel


from inference import inference
from finetune_utils.metric import compute_mcd
from dataset import fetch_dataset, create_manifest, split_data

DATA_DIR = "./data"

def load_data():
    
    fetch_dataset(
        repo_id="syarief-mulyadi/pbl-tts-dataset",
        filename="sundanese/su_id_male_01596.zip",
        repo_type="dataset",
        download_dir=DATA_DIR,
    )
    
    full_transcript_path = Path(DATA_DIR) / "line_index.tsv"
    full_manifest_path = Path(DATA_DIR) / "all_manifest.json"
    train_manifest_path = Path(DATA_DIR) / "train_manifest.json"
    val_manifest_path = Path(DATA_DIR) / "val_manifest.json"

    create_manifest(
        data_dir=DATA_DIR,
        transcript_path=str(full_transcript_path),
        manifest_path=str(full_manifest_path)
    )
    
    split_data(
        full_manifest_path=str(full_manifest_path),
        train_manifest_path=str(train_manifest_path),
        val_manifest_path=str(val_manifest_path),
        eval_size=10,
        seed=42
    )

@hydra.main(version_base=None, config_path="config", config_name="inference")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    # Load and verify dataset
    # TODO: add logic to load custom dataset
    data_dir = Path(DATA_DIR)
    wavs_dir = data_dir / "wavs"
    train_manifest = data_dir / "train_manifest.json"
    val_manifest = data_dir / "val_manifest.json"

    data_is_valid = (
        wavs_dir.is_dir() and           # `data/wavs/` directory exists
        any(wavs_dir.iterdir()) and     # `data/wavs/` is not empty
        train_manifest.is_file() and    # `data/train_manifest.json` exists
        val_manifest.is_file()          # `data/val_manifest.json` exists
    )

    if not data_is_valid:
        if data_dir.exists():
            shutil.rmtree(data_dir)
        load_data()

    val_dataset = TTSDataset(manifest_filepath=val_manifest)

    # Load model
    e2e_model = None
    generator = None
    vocoder   = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.model_choice == "vits":
        vits_cfg = cfg.vits_config
        if vits_cfg.load_method == "pretrained":
            e2e_model = VitsModel.from_pretrained(vits_cfg.pretrained_model).to(device)
        elif vits_cfg.load_method == "checkpoint":
            e2e_model = VitsModel.restore_from(vits_cfg.checkpoint_path).to(device)
        e2e_model.eval()

    elif cfg.model_choice == "fastpitch_hifigan":
        comp_cfg = cfg.components_config
        
        if comp_cfg.generator.load_method == "pretrained":
            generator = FastPitchModel.from_pretrained(comp_cfg.generator.pretrained_model).to(device)
        elif comp_cfg.generator.load_method == "checkpoint":
            generator = FastPitchModel.restore_from(comp_cfg.generator.checkpoint_path).to(device)
        
        if comp_cfg.vocoder.load_method == "pretrained":
            vocoder = HifiGanModel.from_pretrained(comp_cfg.vocoder.pretrained_model).to(device)
        elif comp_cfg.vocoder.load_method == "checkpoint":
            vocoder = HifiGanModel.restore_from(comp_cfg.vocoder.checkpoint_path).to(device)
            
        generator.eval()
        vocoder.eval()
    
    else:
        raise ValueError(f"Unknown model_choice in config: {cfg.model_choice}")

    # Inference process
    audio_preds, scores = inference(
        batches=val_dataset,
        sample_rate=cfg.sample_rate,
        score_fn=compute_mcd,
        e2e_model=e2e_model,
        generator=generator,
        vocoder=vocoder,
    )

    # Save inference results
    output_dir = Path.cwd()
    results_dir = output_dir / "inference_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    total_score = 0.0
    score_save_path = results_dir / "_scores.txt"

    with open(score_save_path, 'w', encoding='utf-8') as f:
        f.write("Sample_Index\tScore\n") 
        
        for i, (audio_data, mcd_tensor) in enumerate(zip(audio_preds, scores)):
            save_path = results_dir / f"pred_sample_{i}.wav"
            sf.write(str(save_path), audio_data, cfg.sample_rate)
            
            mcd_value = mcd_tensor.item()
            f.write(f"{i}\t{mcd_value:.6f}\n")
            total_score += mcd_value

        avg_mcd = total_score / len(scores)
        f.write(f"\nAverage_Score: {avg_mcd:.6f}\n")
        
if __name__ == "__main__":
    main()
