# Rotwild-ID: deer-face-embed

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19793946.svg)](https://doi.org/10.5281/zenodo.19793946)

Metric learning model for individual red deer (*Cervus elaphus*) identification from facial images. Trains a ViT or CNN backbone with metric-learning losses (default: triplet) to produce L2-normalized face embeddings, enabling re-identification across images of deer.

Trained and evaluated on the [RotwildID Faces](https://www.kaggle.com/datasets/jonaschu/rotwildid-faces) dataset (59 individuals, ~1,000 face images, three face chip variants). Experiment tracking via [MLflow](https://mlflow.org/), configuration via [Hydra](https://hydra.cc/) + Pydantic — models, losses, augmentations, and schedulers are all swappable from the CLI.

---

## Benchmark Results

Best result per backbone on RotwildID Faces (mask chips, 80/20 train/val split):

| Backbone | P@1 (val) | mAP (val) | ROC-AUC | Freeze strategy |
|---|---|---|---|---|
| InceptionNext-tiny | 0.618 | 0.617 | 0.957 | Groups [2,3] trainable |
| **InceptionNext-base** | **0.817** | **0.804** | **0.991** | g3 → g2 (transition ep. 60) |

Key sweep findings:

| Sweep | Best config | P@1 (val) |
|---|---|---|
| Freeze strategy | g3 then g2 @ ep 60 | 0.645 |
| Augmentation | Slightly conservative (flip, photometric, blur, light erase) | 0.789 |
| Training mechanism | MPerClass sampler + CosineAnnealingWarmRestarts | 0.711 |
| Embedding size | 512 | 0.768 |

Top single run (AdamW, intermediate_size=1024, embedding_size=128, seed=42): **P@1=0.817, mAP=0.804, ROC-AUC=0.991**

Sweep numbers above are best-in-sweep with all other config left at defaults — they isolate the effect of one axis and are not directly comparable to the top-line result.

---

## Dataset

**Kaggle**: [`jonaschu/rotwildid-faces`](https://www.kaggle.com/datasets/jonaschu/rotwildid-faces)

Three face chip variants are available as subdirectories:
- `mask/` — mask-based face extraction (best benchmark results)
- `landmark_affine/` — affine-aligned chips using facial landmarks
- `landmark_crop/` — bounding-box crop chips

All variants follow the [WildlifeDatasets](https://github.com/WildlifeDatasets/wildlife-datasets) format. The `KaggleDatasetLoader` downloads and caches the dataset automatically on first run.

Set the `KAGGLE_API_TOKEN` environment variable before training (see [Kaggle API docs](https://www.kaggle.com/docs/api)).

For training on your own data with `LocalDatasetLoader`, see [docs/custom_dataset.md](docs/custom_dataset.md).

---

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then pick the workflow that matches your hardware. Training without a CUDA GPU is impractical for full runs — CPU is for smoke tests only.

```bash
# Recommended: NVIDIA GPU
uv sync --extra cu128
source .venv/bin/activate
# from now on, call the entry point directly:
deer-face-embed

# CPU only (smoke tests, not full training)
uv sync --extra cpu
source .venv/bin/activate
deer-face-embed
```

> **Note:** activating the venv puts `deer-face-embed` on `PATH` against the already-synced environment, so you don't need to repeat `--extra cu128` on every call. The equivalent without activation is `uv run --extra cu128 deer-face-embed …`.

---

## Quick Start

> **Note:** Need a Kaggle API key? See the [Kaggle API docs](https://www.kaggle.com/docs/api).

```bash
# 1. Set Kaggle credentials
export KAGGLE_API_TOKEN=your_api_token

# 2. Start MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5001
# for using mlflow in Docker, see MLFlow Section

# 3. Train with default config (InceptionNext-base, mask chips)
deer-face-embed

# 4. View results at http://localhost:5001
```

> **Note:** if you skip step 2, the run is still logged to the local `./mlruns` directory — you just won't see the live UI.

---

## Configuration

Configuration uses [Hydra](https://hydra.cc/) with Pydantic validation.

### Discovering options

```bash
# List all parameters and their default values
deer-face-embed --help

# Hydra basics: overrides, multirun, config groups (useful if you're new to Hydra)
deer-face-embed --hydra-help
```

### Overriding parameters

Override any parameter from the CLI:

```bash
# Change backbone
deer-face-embed embedder_config.KIND=VitBase

# Change number of epochs
deer-face-embed training_config.num_epochs=100

# Use landmark_affine face chips
deer-face-embed dataset_loader.face_chip_method=landmark_affine

# Use local dataset instead of Kaggle
deer-face-embed dataset_loader.KIND=LocalDatasetLoader dataset_loader.dataset_location=./data/extracted
```

### Available Models

| KIND | Architecture |
|---|---|
| `inception_next` | InceptionNeXt (default: base) |
| `ResNet` | ResNet-50d |
| `Swin` | Swin Transformer |
| `MegaDescriptor` | MegaDescriptor-L |
| `VitDino` | ViT-S/16 with DINO weights |
| `VitBase` | ViT-B/16 |
| `DenseVit` | DenseNet121 + ViT hybrid |

All models use a pretrained [timm](https://github.com/huggingface/pytorch-image-models) backbone with a custom embedding head (linear → ReLU → dropout → linear → BatchNorm → L2 normalize).

### Backbone Freezing

Two-phase freezing via `embedder_config.backbone_freeze`:

```yaml
backbone_freeze:
  initial:
    trainable_groups: [3]    # only group 3 (last) trainable at start
  transition_epoch: 60       # switch to final phase after epoch 60
  final:
    trainable_groups: [2, 3] # groups 2 and 3 trainable after transition
```

Omit `backbone_freeze` entirely for a fully trainable backbone. Set `final: null` to unfreeze all groups after transition.

### Hyperparameter Sweeps

```bash
deer-face-embed --config-name=base_model_sweep --multirun    # Compare backbones
deer-face-embed --config-name=freeze_sweep --multirun        # Compare freeze strategies
deer-face-embed --config-name=augmentation_sweep --multirun  # Compare augmentations
deer-face-embed --config-name=face_chip_sweep --multirun     # Compare face chip types
```

---

## MLflow

### Local server

```bash
uv run mlflow ui --backend-store-uri ./mlruns --port 5001
```

By default, runs and artifacts are written to `./mlruns` relative to the current working directory. If you start the UI from a different directory than your training runs, the UI will look in the wrong place and experiments appear missing. Either start `mlflow ui` from the project root, or point `--backend-store-uri` at an absolute path:

```bash
uv run mlflow ui --backend-store-uri /absolute/path/to/face-embedding-model/mlruns --port 5001
```

### Docker

```bash
cd mlflow && docker compose up -d
# Access at http://localhost:5001
```

When using the Docker-based MLflow server, the client must log to the server via HTTP instead of writing to the local `./mlruns` folder. Enable this by adding an `mlflow_service` block in [config/job/training.yaml](config/job/training.yaml) to overwrite defaults.

```yaml
mlflow_service:
  tracking_uri: http://127.0.0.1:5001
  registry_uri: http://127.0.0.1:5001
```

Without this, training will log to the local `./mlruns` directory, bypassing the Docker server.

---

## Citation

```bibtex
@software{deer_face_embed,
  author    = {Schulze Buschhoff, Jonathan},
  title     = {Rotwild-ID deer-face-embed: Metric learning for red deer re-identification},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19793946},
  url       = {https://github.com/rotwild-id/deer-face-embed}
}
```

---

## Acknowledgements

This project builds on pretrained backbones and open-source tooling from the broader ML and wildlife re-identification community.

### Backbones

All backbones are loaded via [timm](https://github.com/huggingface/pytorch-image-models) with their original pretrained weights.

- **InceptionNeXt** (default) — Yu, W., Zhou, P., Yan, S., & Wang, X. (2024). *InceptionNeXt: When Inception Meets ConvNeXt.* CVPR 2024. [arXiv:2303.16900](https://arxiv.org/abs/2303.16900)
- **ResNet-50d** — He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385). The `-d` variant follows He, T. et al. (2018). *Bag of Tricks for Image Classification with Convolutional Neural Networks.* [arXiv:1812.01187](https://arxiv.org/abs/1812.01187)
- **Swin Transformer** — Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- **MegaDescriptor** — Čermák, V., Picek, L., Adam, L., & Papafitsoros, K. (2024). *WildlifeDatasets: An Open-Source Toolkit for Animal Re-Identification.* WACV 2024. Weights: [`BVRA/MegaDescriptor-L-224`](https://huggingface.co/BVRA/MegaDescriptor-L-224)
- **ViT-B/16** — Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- **ViT-DINO** — Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). *Emerging Properties in Self-Supervised Vision Transformers.* ICCV 2021. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)
- **DenseNet-121** (used inside `DenseVit`) — Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017). *Densely Connected Convolutional Networks.* CVPR 2017. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)
- **DenseVit hybrid architecture** — adapted from He, K., Zhao, F., Wang, Z., Cao, X., Yan, F., Liu, J., & Wang, J. (2023). *Animal re-identification algorithm for posture diversity.* Ecological Informatics. [doi:10.1016/j.ecoinf.2023.102334](https://doi.org/10.1016/j.ecoinf.2023.102334)

### Tooling

- [**timm**](https://github.com/huggingface/pytorch-image-models) — Wightman, R. (2019). *PyTorch Image Models.*
- [**pytorch-metric-learning**](https://github.com/KevinMusgrave/pytorch-metric-learning) — Musgrave, K., Belongie, S., & Lim, S.-N. (2020). *PyTorch Metric Learning.* [arXiv:2008.09164](https://arxiv.org/abs/2008.09164)
- [**wildlife-tools**](https://github.com/WildlifeDatasets/wildlife-tools) and [**wildlife-datasets**](https://github.com/WildlifeDatasets/wildlife-datasets) — Čermák et al. (2024), as cited above.
- [**Hydra**](https://hydra.cc/) — Yadan, O. (2019). *Hydra — A framework for elegantly configuring complex applications.* Facebook AI Research.
- [**MLflow**](https://mlflow.org/) — Zaharia, M. et al. (2018). *Accelerating the Machine Learning Lifecycle with MLflow.* IEEE Data Eng. Bull.


---

## License

Apache 2.0 — Copyright © Landesjagdverband Schleswig-Holstein e.V.
