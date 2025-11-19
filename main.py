from random import sample
import shutil
from pathlib import Path

import torch
import hydra
import soundfile as sf
from omegaconf import DictConfig, OmegaConf
from nemo.collections.tts.models.vits import VitsModel
from nemo.collections.tts.data.dataset import TTSDataset
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import EnglishPhonemesTokenizer
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from nemo.collections.tts.g2p.models.en_us_arpabet import EnglishG2p


from inference import inference
from finetune_utils.metric import compute_mcd
from dataset import fetch_dataset, create_manifest, split_data

DATA_DIR = Path("./data")

def load_data(repo_id: str, filename: str, data_name: str) -> None:
    
    fetch_dataset(
        repo_id=repo_id,
        filename=filename,
        download_dir=DATA_DIR / data_name,
    )
    
    full_transcript_path = DATA_DIR / data_name / "line_index.tsv"
    full_manifest_path = DATA_DIR / data_name / "all_manifest.json"
    train_manifest_path = DATA_DIR / data_name / "train_manifest.json"
    val_manifest_path = DATA_DIR / data_name / "val_manifest.json"

    create_manifest(
        data_dir=DATA_DIR / data_name,
        transcript_path=str(full_transcript_path),
        manifest_path=str(full_manifest_path)
    )
    
    split_data(
        full_manifest_path=str(full_manifest_path),
        train_manifest_path=str(train_manifest_path),
        eval_manifest_path=str(val_manifest_path),
        eval_size=10,
        seed=42
    )

@hydra.main(version_base=None, config_path="config", config_name="inference")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    method = cfg.get("method", None)

    # Load and verify dataset
    data_working_dir = DATA_DIR / cfg.dataset.data_name
    wavs_dir = data_working_dir / "wavs"
    train_manifest = data_working_dir / "train_manifest.json"
    val_manifest = data_working_dir / "val_manifest.json"

    # TODO: Check valid cached data exists fail
    data_is_valid = (
        wavs_dir.is_dir() and           # `data/{data_name}/wavs/` directory exists
        any(wavs_dir.iterdir()) and     # `data/{data_name}/wavs/` is not empty
        train_manifest.is_file() and    # `data/{data_name}/train_manifest.json` exists
        val_manifest.is_file()          # `data/{data_name}/val_manifest.json` exists
    )

    if not data_is_valid:
        if data_working_dir.exists():
            shutil.rmtree(data_working_dir)
        load_data(cfg.dataset.repo_id, cfg.dataset.filename, cfg.dataset.data_name)

    val_dataset = TTSDataset(manifest_filepath=[val_manifest], sample_rate=22050, text_tokenizer=EnglishPhonemesTokenizer(g2p=EnglishG2p(phoneme_dict="finetune_utils/cmudict-0.7b_nv22.10", heteronyms="finetune_utils/heteronyms-052722")))

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
            vits_ckpt_path = vits_cfg.checkpoint_path
            if vits_ckpt_path is None:
                raise ValueError("VITS checkpoint path must be provided when load_method is 'checkpoint'.")
            if str(vits_ckpt_path).endswith(".nemo"):
                e2e_model = VitsModel.restore_from(vits_ckpt_path).to(device)
            else:
                e2e_model = VitsModel.load_from_checkpoint(vits_ckpt_path).to(device)
        e2e_model.eval()

    elif cfg.model_choice == "fastpitch_hifigan":
        comp_cfg = cfg.components_config
        
        if comp_cfg.generator.load_method == "pretrained":
            generator = FastPitchModel.from_pretrained(comp_cfg.generator.pretrained_model).to(device)
        elif comp_cfg.generator.load_method == "checkpoint":
            gen_ckpt_path = comp_cfg.generator.checkpoint_path
            if gen_ckpt_path is None:
                raise ValueError("Generator checkpoint path must be provided when load_method is 'checkpoint'.")
            if str(gen_ckpt_path).endswith(".nemo"):
                generator = FastPitchModel.restore_from(gen_ckpt_path).to(device)
            else:
                generator = FastPitchModel.load_from_checkpoint(gen_ckpt_path).to(device)
        
        if comp_cfg.vocoder.load_method == "pretrained":
            vocoder = HifiGanModel.from_pretrained(comp_cfg.vocoder.pretrained_model).to(device)
        elif comp_cfg.vocoder.load_method == "checkpoint":
            voc_ckpt_path = comp_cfg.vocoder.checkpoint_path
            if voc_ckpt_path is None:
                raise ValueError("Vocoder checkpoint path must be provided when load_method is 'checkpoint'.")
            if str(voc_ckpt_path).endswith(".nemo"):
                vocoder = HifiGanModel.restore_from(voc_ckpt_path).to(device)
            else:
                vocoder = HifiGanModel.load_from_checkpoint(voc_ckpt_path).to(device)
            
        generator.eval()
        vocoder.eval()
    
    else:
        raise ValueError(f"Unknown model_choice in config: {cfg.model_choice}")

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
    method_suffix = f"_{method}" if method else ""
    results_dir = output_dir / "results" / f'{cfg.model_choice}_{cfg.dataset.data_name}{method_suffix}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    total_score = 0.0
    score_save_path = results_dir / "_scores.txt"

    with open(score_save_path, "w", encoding="utf-8") as f:
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
