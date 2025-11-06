# Praktikum 2 PBL: Text-to-Speech

## Installation

Install `uv` package manager.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dependencies

```bash
uv sync
```

## Usage

### Pre-trained Experiment

You can run inference using default pretrained models by overriding the config on the command line.

```bash
uv run main.py \
    model_choice=fastpitch_hifigan \
    components_config.generator.load_method=pretrained \
    components_config.generator.pretrained_model="tts_en_fastpitch" \
    components_config.vocoder.load_method=pretrained \
    components_config.vocoder.pretrained_model="tts_en_hifigan"
```

or this, for VITS model.

```bash
uv run main.py \
    model_choice=vits \
    vits_config.load_method=pretrained \
    vits_config.pretrained_model="tts_en_vits"
```

### Fine-tuned Experiment

First, edit the `.sh` script(s) in the `scripts/` directory (e.g., `finetune_vits.sh`) to point to the base `.nemo` model you want to finetune from (the `+init_from_nemo_model={...}` line).

Next, make the scripts executable, then run the desired fine-tuning script.

```bash
chmod +x scripts/finetune_{vits/fastpitch/hifigan}.sh
./scripts/finetune_{vits/fastpitch/hifigan}.sh
```

After finetuning, you can run inference by pointing to your new checkpoint files.

```bash
uv run main.py \
    model_choice=fastpitch_hifigan \
    components_config.generator.load_method=checkpoint \
    components_config.generator.checkpoint_path="/path/to/your/finetuned_fastpitch.nemo" \
    components_config.vocoder.load_method=checkpoint \
    components_config.vocoder.checkpoint_path="/path/to/your/finetuned_hifigan.nemo"
```

or this, for VITS model.

```bash
uv run main.py \
    model_choice=vits \
    vits_config.load_method=checkpoint \
    vits_config.checkpoint_path="/path/to/your/finetuned_vits.nemo"
```
