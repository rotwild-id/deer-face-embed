# Project: deer-face-embed

## Overview
Deep learning project for wildlife identification using face embeddings. Trains models to identify individual red deer from facial images using metric learning (triplet loss, ViT/CNN backbones).

## Stack
- **Language**: Python 3.12
- **Framework**: PyTorch, TorchVision
- **Experiment Tracking**: MLflow
- **Configuration**: Hydra + Pydantic
- **Package Manager**: uv
- **Key Libraries**: pytorch-metric-learning, timm, wildlife-tools, wildlife-datasets, faiss-cpu, loguru, kaggle

## Project Structure
```
src/deer_face_embed/
├── core/
│   ├── models/          # base.py (BaseEmbedder, freeze config), CNNBasedModels.py, VitBasedModels.py
│   ├── trainer.py       # Trainer + training component configs (loss, miner, optimizer, scheduler)
│   ├── evaluation/      # MetricsCalculator, ClosedSetEvaluator
│   └── augmentation.py  # Augmentation configs (pydantic discriminated unions)
├── jobs/                # Hydra job definitions (training)
├── io/                  # Data loading, Kaggle integration, wildlife_dataset
└── main.py              # CLI entry point
config/
├── default.yaml         # Top-level default (loads job/training.yaml)
├── job/training.yaml    # Default training job config
├── augmentation/        # baseline.yaml, conservative.yaml, aggressive.yaml, none.yaml
├── freeze/              # Freeze strategy presets
├── base_model_sweep.yaml
├── freeze_sweep.yaml
├── augmentation_sweep.yaml
└── face_chip_sweep.yaml
```

## Workflow
1. **Install dependencies**: `uv sync --extra cu128` (GPU) or `uv sync --extra cpu` (CPU only)
2. **Activate venv**: `source .venv/bin/activate` (so `deer-face-embed` is on `PATH` and `uv run` doesn't re-resolve every call)
3. **Set Kaggle credentials**: `export KAGGLE_API_TOKEN=...`
4. **Start MLflow UI**: `mlflow ui --backend-store-uri "./mlruns" --port 5001` or `cd mlflow && docker compose up -d`
5. **Train model**: `deer-face-embed`
6. **Hyperparameter sweep**: `deer-face-embed --config-name=base_model_sweep --multirun`

## Configuration System

### Architecture
- **Hydra** loads YAML configs and provides CLI overrides
- **Pydantic** validates all configs with discriminated unions (`KIND` field selects implementation)
- Config classes live alongside their implementations (e.g., `TripletLossConfig` in `trainer.py`)

### Config Hierarchy
- `config/default.yaml` → loads `config/job/training.yaml`
- Sweep configs override the base with `defaults: [job: training, _self_]`
- CLI overrides: `deer-face-embed training_config.num_epochs=200`

### Dataset Config (`dataset_loader`)
- `KIND: "KaggleDatasetLoader"` (default) — downloads from Kaggle automatically
- `KIND: "LocalDatasetLoader"` — for users with local data
- `face_chip_method`: `"mask"` | `"landmark_affine"` | `"landmark_crop"`

### Model Config (`embedder_config`)
Required: `KIND`, `pretrained`, `embedding_size`
Optional: `batch_size` (32), `intermediate_size` (1024), `dropout_rate` (0.3), `backbone_freeze` (null)

Available KINDs: `inception_next`, `ResNet`, `Swin`, `MegaDescriptor`, `VitDino`, `VitBase`, `DenseVit`

### Backbone Freezing (`embedder_config.backbone_freeze`)
Two-phase freezing system. Each model defines layer groups via `get_layer_groups()`.
- CNN models (ResNet, InceptionNext): groups = architectural stages (0-3)
- Swin/MegaDescriptor: groups = `layers.0`..`layers.3` (0-3)
- ViT models: 12 blocks split into 4 groups of 3 (0-3)
- DenseVit: groups 0-3 (DenseNet blocks), 4-7 (ViT blocks)

### Training Config (`training_config`)
Required: `num_epochs`, `loss` (KIND), `miner` (KIND), `optimizer` (KIND)

**Loss KINDs**: `TripletLoss`
**Miner KINDs**: `TripletMarginMiner`, `MultiSimilarityMiner`
**Optimizer KINDs**: `Adam`, `AdamW` (with `backbone_lr_factor` for differential LR)

Optional: `lr_scheduler`, `warmup`, `early_stopping`, `sampler`, `gradient_clip_val`

### Augmentation Config
Available KINDs: `RandomPhotometricDistort`, `RandomHorizontalFlip`, `RandomGrayscale`,
`RandomAdjustSharpness`, `RandomGaussianBlur`, `RandomRotation`, `RandomErasing`,
`RandomResizedCrop`, `ColorJitter`, `RandomGaussianNoise`, `Resize`

## Code Conventions

### Architecture
- **Model Pattern**: timm backbone → embedding head → L2 normalization
- **Config Pattern**: Pydantic discriminated unions (`KIND` field) for polymorphic config
- **Logging**: loguru for application logging, MLflow for experiment tracking

### Python Style
- **Formatting**: Ruff (88-char line length, Google docstring convention)
- **Type Hints**: Use throughout (pydantic for configs, typing for functions)

### ML-Specific Conventions
- **Metrics**: Log to MLflow: loss curves, ROC, precision@1, mAP, NMI
- **Reproducibility**: Set seeds, log hyperparameters, save model artifacts to MLflow
- **Best model**: Tracked by precision_at_1.val, saved to MLflow once after training

## Critical Rules

### MLflow Integration
1. Default tracking: `./mlruns` (local). No remote server required.
2. Log all hyperparameters and metrics to MLflow (not just stdout)
3. Best model artifact is logged once after training completes (not per-epoch)

### Config Management
- Never hardcode paths or hyperparameters — use Hydra configs
- Default configs in `config/`, job-specific in `config/job/`

### Data Handling
- Wildlife datasets use the wildlife-datasets library (`WildlifeDataset` format)
- Kaggle credentials required: `KAGGLE_API_TOKEN` env var
- Face chips preprocessed consistently (size, normalization)

### GPU/CUDA
- Designed for consumer hardware (NVIDIA RTX 3080, 12 GB VRAM)
- CPU PyTorch installed by default; override for GPU training
- Backbone freezing reduces VRAM by limiting trainable params and optimizer state

### Version Control
- **Commit**: Frequent atomic commits after each logical change
- **Never commit**: data/, mlruns/, .env files, model artifacts

## Commands Reference

| Command | Purpose |
|---------|---------|
| `uv sync --extra cu128` / `uv sync --extra cpu` | Install/update dependencies (GPU / CPU) |
| `source .venv/bin/activate` | Activate venv so `deer-face-embed` is on `PATH` |
| `deer-face-embed` | Train model (default config) |
| `deer-face-embed --help` | List all parameters and their defaults |
| `deer-face-embed --hydra-help` | Hydra basics: overrides, multirun, config groups |
| `deer-face-embed --config-name=base_model_sweep --multirun` | Backbone comparison sweep |
| `deer-face-embed --config-name=freeze_sweep --multirun` | Freeze strategy sweep |
| `deer-face-embed --config-name=augmentation_sweep --multirun` | Augmentation sweep |
| `deer-face-embed --config-name=face_chip_sweep --multirun` | Face chip type sweep |
| `mlflow ui --backend-store-uri "./mlruns" --port 5001` | Start MLflow UI locally |
| `cd mlflow && docker compose up -d` | Start MLflow server via Docker |
| `pytest` | Run tests |
| `ruff check .` | Lint codebase |
| `ruff format .` | Auto-format code |

### Development commands
| Command | Purpose |
|---------|---------|
| `make lint` | lint |
| `make typecheck` | typecheck |
| `make format` | run ruff formatter |
| `make format-check` | Check formatting without modifying files |
| `make fix` | ruff linter + formatter with auto-fixes |
| `make check` | lint + format-check + typecheck |
