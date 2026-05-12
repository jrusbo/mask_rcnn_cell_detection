 # Mask R-CNN Cell Detection

This repository contains a cell instance-segmentation pipeline built around a Mask R-CNN model with a ResNet-50-FPN backbone and deformable convolutions in the deeper backbone stages.

The project supports four main workflows:

- preprocessing raw TIFF masks into a COCO-style tiled dataset
- training a detector/segmenter on the generated COCO data
- evaluating a trained checkpoint on the validation split
- running inference on the test release images and exporting a submission JSON

## Repository layout

```text
src/
  model.py            # Model factory and DCNv2 wrapper
  preprocessing.py    # Raw-to-COCO dataset conversion
  train.py            # Training loop
  predict.py          # Test-set inference / submission writer
  evaluate_checkpoint.py # Validation evaluation script
  test_dataset.py     # Sanity-check visualizer for generated data
  tiling_utils.py     # Framework-agnostic tiling helpers
datasets/
  train/              # Raw training folders (expected input)
  coco_format/        # Generated COCO dataset and preview images
checkpoints/          # Saved training runs
```

## Requirements

- Python 3.13 or newer
- `uv`
- PyTorch with CUDA support if you want to train or evaluate on a GPU

The project is configured in `pyproject.toml` and uses CUDA 13.0 wheels for `torch`, `torchvision`, and `torchaudio`.

## Installation

1) Create a virtual environment and sync the main dependencies:

```powershell
uv venv --python 3.13
uv sync
```

2) (Optional) Install developer tools (lightweight linter `ruff`) via uv extras:

```powershell
# Installs the optional 'dev' extra defined in pyproject.toml
uv sync --extra dev
```

Using `ruff` (recommended)

`ruff` is a fast, lightweight linter and auto-formatter. After installing the `dev` extra, run ruff directly:

```bash
# Check the source tree
uv run ruff check src

# Auto-fix trivial issues (rewrites files)
uv run ruff check --fix src

# Format only (equivalent to black-style formatting)
uv run ruff format src
```

Ruff reads configuration from `pyproject.toml` when present. It's preferred over heavy tools like `pylint` for speed.

3) Set up Weights & Biases (required to log training runs):

```powershell
uv run wandb login
# Paste your API key when prompted
```

## Prepare the dataset

The preprocessing pipeline expects a raw folder structure containing per-sample directories with files like:

- `image.tif`
- `class1.tif`
- `class2.tif`
- `class3.tif`
- `class4.tif`

By default, `src/preprocessing.py` points `RAW_DATA_DIR` to a Kaggle path. Update that constant to your local training directory before running the script.

Run preprocessing:

```powershell
uv run src/preprocessing.py
```

This generates:

- `datasets/coco_format/train.json`
- `datasets/coco_format/val.json`
- tiled images under `datasets/coco_format/images/train` and `datasets/coco_format/images/val`

## Train the model

Training uses Albumentations for augmentation, COCO annotations for the dataset, and Weights & Biases for logging.

```powershell
uv run src/train.py
```

Optional resume example:

```powershell
uv run src/train.py --resume checkpoints\20260507_144335\latest_model.pth
```

Optional custom run name:

```powershell
uv run src/train.py --run_name my_experiment
```

Training outputs are written to `checkpoints/<run_name>/`.

### Notes

- The training script is designed for GPU execution.
- Distributed training is enabled automatically when more than one CUDA device is available.
- The model uses a focal-loss patch on the classification branch to help with class imbalance.

## Evaluate a checkpoint

Use the validation evaluation script to compute AP50 metrics for a trained checkpoint.

```powershell
uv run src/evaluate_checkpoint.py
```

Before running it, update `MODEL_WEIGHTS` in `src/evaluate_checkpoint.py` to point to the checkpoint you want to evaluate.

## Run inference

Generate a submission JSON for the test release images:

```powershell
uv run src/predict.py --model checkpoints\20260507_144335\best_model.pth
```

This writes `test-results.json` in the repository root.

## Sanity-check the generated dataset

If you want to inspect the generated COCO annotations visually, run:

```powershell
uv run src/test_dataset.py
```

This saves preview PNGs into `datasets/coco_format/` and opens a window when OpenCV GUI support is available.

## Important configuration knobs

- `src/preprocessing.py`
  - `RAW_DATA_DIR`: location of the raw training folders
  - `PATCH_SIZE`, `OVERLAP`: tiling settings
  - `NUM_CLASSES`: number of foreground cell categories
- `src/train.py`
  - `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`: training hyperparameters
  - `MIN_SIZE`, `MAX_SIZE`: image resizing limits used by the detector
- `src/predict.py`
  - `SCORE_THRESHOLD`, `MASK_THRESHOLD`: inference filtering thresholds

## Expected outputs

- `datasets/coco_format/*.json`: generated COCO annotations
- `datasets/coco_format/images/*`: tiled training/validation images
- `checkpoints/<run_name>/*.pth`: saved checkpoints
- `test-results.json`: inference output in submission format


