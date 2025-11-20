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
from dataset import (
    fetch_dataset,
    create_manifest,
    split_data,
    download_and_extract_tgz_dataset,
    convert_language_subset_to_wav,
    create_manifest_from_librivox,
    split_manifest_entries,
)

DATA_DIR = Path("./data")


def load_data(cfg: DictConfig) -> None:
    dataset_cfg = cfg.dataset
    variant = (dataset_cfg.get("variant") or "sundanese").lower()

    if variant == "sundanese":
        _prepare_sundanese_dataset(dataset_cfg)
    elif variant == "indonesian":
        _prepare_indonesian_dataset(cfg, dataset_cfg)
    else:
        raise ValueError(f"Unsupported dataset.variant '{variant}'.")


def _prepare_sundanese_dataset(dataset_cfg: DictConfig) -> None:
    data_dir = DATA_DIR / dataset_cfg.data_name
    data_dir.mkdir(parents=True, exist_ok=True)

    fetch_dataset(
        repo_id=dataset_cfg.repo_id,
        filename=dataset_cfg.filename,
        download_dir=data_dir,
    )

    transcript_filename = dataset_cfg.get("transcript_filename") or "line_index.tsv"
    full_transcript_path = data_dir / transcript_filename
    full_manifest_path = data_dir / "all_manifest.json"
    train_manifest_path = data_dir / "train_manifest.json"
    val_manifest_path = data_dir / "val_manifest.json"

    create_manifest(
        data_dir=data_dir,
        transcript_path=str(full_transcript_path),
        manifest_path=str(full_manifest_path),
    )

    split_cfg = dataset_cfg.get("split") or {}
    eval_size = split_cfg.get("eval_size", 10)
    seed = split_cfg.get("seed", 42)

    split_data(
        full_manifest_path=str(full_manifest_path),
        train_manifest_path=str(train_manifest_path),
        eval_manifest_path=str(val_manifest_path),
        eval_size=eval_size,
        seed=seed,
    )


def _prepare_indonesian_dataset(cfg: DictConfig, dataset_cfg: DictConfig) -> None:
    data_dir = DATA_DIR / dataset_cfg.data_name
    data_dir.mkdir(parents=True, exist_ok=True)

    extract_root = download_and_extract_tgz_dataset(
        repo_id=dataset_cfg.repo_id,
        filename=dataset_cfg.filename,
        download_dir=str(data_dir),
    )

    extracted_subdir = dataset_cfg.get("extracted_subdir") or "librivox-indonesia"
    raw_dataset_dir = Path(extract_root) / extracted_subdir
    if not raw_dataset_dir.exists():
        raise FileNotFoundError(f"Expected extracted dataset at '{raw_dataset_dir}' not found.")

    metadata_filename = dataset_cfg.get("metadata_filename") or "metadata_test.csv"
    metadata_csv = raw_dataset_dir / metadata_filename
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV '{metadata_csv}' not found.")

    language = dataset_cfg.get("language") or "indonesian"
    split_subdir = dataset_cfg.get("split_subdir") or "test"
    wav_root = data_dir / "wavs"

    convert_language_subset_to_wav(
        dataset_root=str(raw_dataset_dir),
        language=language,
        split_subdir=split_subdir,
        output_dir=str(wav_root),
        sample_rate=cfg.sample_rate,
    )

    mel_dir = data_dir / "mels"
    manifest_path = data_dir / "all_manifest.json"
    manifest_entries = create_manifest_from_librivox(
        metadata_csv_path=str(metadata_csv),
        wav_root=str(wav_root),
        manifest_path=str(manifest_path),
        language=language,
        mel_dir=str(mel_dir),
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.n_window_stride,
        win_length=cfg.n_window_size,
        n_mels=cfg.n_mel_channels,
    )

    split_cfg = dataset_cfg.get("split") or {}
    eval_size = split_cfg.get("eval_size", 10)
    train_size = split_cfg.get("train_size")
    seed = split_cfg.get("seed", 42)

    if eval_size is None:
        raise ValueError("dataset.split.eval_size must be provided for the Indonesian dataset.")
    if train_size is None:
        if len(manifest_entries) <= eval_size:
            raise ValueError("Not enough samples to derive a train split.")
        train_size = len(manifest_entries) - eval_size

    train_manifest_path = data_dir / "train_manifest.json"
    val_manifest_path = data_dir / "val_manifest.json"

    split_manifest_entries(
        entries=manifest_entries,
        train_manifest_path=str(train_manifest_path),
        eval_manifest_path=str(val_manifest_path),
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
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
        load_data(cfg)

    val_dataset = TTSDataset(
        manifest_filepath=[val_manifest],
        sample_rate=cfg.sample_rate,
        text_tokenizer=EnglishPhonemesTokenizer(
            g2p=EnglishG2p(
                phoneme_dict="finetune_utils/cmudict-0.7b_nv22.10",
                heteronyms="finetune_utils/heteronyms-052722",
            )
        ),
    )

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
