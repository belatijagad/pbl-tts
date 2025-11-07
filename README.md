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

You can run inference using default pretrained models by overriding the config on the command line.

```bash
./scripts/inference/vits_pt_indo.sh
```

For fine-tuned experiment, do run the fine-tune script first before doing the inference.

```bash
./scripts/finetune/vits.sh
./scripts/inference/vits_ft_indo.sh
```
