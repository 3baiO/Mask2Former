# Boundary-Aware Mask2Former

This repository is a research-oriented Mask2Former extension for Cityscapes semantic segmentation. It keeps the original Mask2Former training pipeline, and adds several boundary-enhancement paths around the pixel decoder and transformer decoder so we can compare boundary-guided variants under a unified codebase.

## What is in this repo

The current project contains three experiment lines:

- `BEFBM`: boundary-enhanced feature bridging with edge supervision
- `BEFBM + BFP`: boundary-aware feature propagation on multi-scale pixel-decoder features
- `BEFBM + HRCA`: boundary-guided decoder refinement with high-resolution cross-attention

The main experiment notes and reproduction details are documented in [BOUNDARY_AWARE_REPRO.md](BOUNDARY_AWARE_REPRO.md).

## Project highlights

- Built on top of Mask2Former for semantic segmentation
- Focused on Cityscapes reproduction and controlled ablation
- Unified launcher for train, resume, background run, and evaluation
- Supports `R50` and `R101` backbones with matched config presets
- Keeps the baseline Mask2Former data flow as intact as possible

## Repository layout

Key files for this project:

- `README.md`: project overview
- `BOUNDARY_AWARE_REPRO.md`: detailed experiment and reproduction guide
- `scripts/train_boundary.sh`: unified launcher for all current variants
- `mask2former/modeling/pixel_decoder/boundary_aware.py`: BEFBM and BFP related modules
- `mask2former/modeling/pixel_decoder/msdeformattn.py`: boundary-aware pixel decoder integration
- `mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py`: decoder-side HRCA modules
- `configs/cityscapes/semantic-segmentation/`: experiment configs for baseline, BFP, and HRCA

## Available experiment presets

### 1. BEFBM baseline

- `configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_boundary.yaml`
- `configs/cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k_boundary.yaml`

### 2. BEFBM + BFP

- `configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_boundary_bfp.yaml`
- `configs/cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k_boundary_bfp.yaml`

### 3. BEFBM + HRCA

- `configs/cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k_boundary_hrca.yaml`
- `configs/cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k_boundary_hrca.yaml`

## Environment

The project is expected to run in the existing `mask2former` conda environment:

```bash
conda activate mask2former
python -c "import torch; print(torch.__version__)"
```

This repository follows the original Mask2Former dependency layout. If you are setting up from scratch, install the base Mask2Former / Detectron2 dependencies first, then run the project-specific experiments from this repo.

## Quick start

### Show launcher help

```bash
scripts/train_boundary.sh --help
```

### Train BEFBM

```bash
scripts/train_boundary.sh --model r50 --variant befbm
```

### Train BFP

```bash
scripts/train_boundary.sh --model r50 --variant bfp
```

### Train HRCA

```bash
scripts/train_boundary.sh --model r50 --variant hrca
```

### Resume training

```bash
scripts/train_boundary.sh --model r50 --variant bfp --resume
```

### Background run

```bash
scripts/train_boundary.sh --model r50 --variant bfp --background
```

### Evaluate a trained model

```bash
scripts/train_boundary.sh \
  --model r50 \
  --variant bfp \
  --mode eval \
  --weights output/model_final.pth
```

## Launcher behavior

The `scripts/train_boundary.sh` helper currently provides:

- `--model <r50|r101>`
- `--variant <befbm|bfp|hrca>`
- `--mode <train|eval>`
- automatic `CUDA_VISIBLE_DEVICES` parsing and `--num-gpus` derivation
- automatic timestamped log creation under `logs/`
- automatic output directory creation under `output/`
- support for resume and background execution

Default preset:

```text
model=r50
variant=bfp
mode=train
gpus=2,3
```

## Main code changes

### Boundary-aware pixel decoder

- boundary extraction and feature bridging are implemented in `mask2former/modeling/pixel_decoder/boundary_aware.py`
- multi-scale boundary outputs are integrated in `mask2former/modeling/pixel_decoder/msdeformattn.py`

### Boundary-guided decoder refinement

- HRCA-related query refinement lives in `mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py`

### Config extensions

The project adds several config groups in `mask2former/config.py`:

- `MODEL.MASK_FORMER.BOUNDARY_AWARE`
- `MODEL.MASK_FORMER.BOUNDARY_PROPAGATION`
- `MODEL.MASK_FORMER.BOUNDARY_DECODER`

## Recommended reading order

If you are new to this repository, read in this order:

1. `README.md`
2. `BOUNDARY_AWARE_REPRO.md`
3. `scripts/train_boundary.sh --help`
4. The matching config file under `configs/cityscapes/semantic-segmentation/`

## Notes

- `output/` and `logs/` are local runtime artifacts and are not meant to be committed
- `FASeg/` is kept outside the main tracked project workflow unless explicitly needed
- the current repository is optimized for experimentation and reproduction rather than packaging

## Acknowledgement

This work is based on the original Mask2Former project:

- Paper: https://arxiv.org/abs/2112.01527
- Original project: https://github.com/facebookresearch/Mask2Former
