"""Configurable trainer for neural network models."""

import abc
import logging
import typing as T
from collections.abc import Sequence


import mlflow
import pandas as pd
import pydantic as pdt
from tqdm import tqdm
import torch
import torch.optim as optim
from torchvision.transforms.v2 import Compose
from torch.utils.data import DataLoader

from pytorch_metric_learning.losses import TripletMarginLoss, BaseMetricLossFunction
from pytorch_metric_learning.miners import (
    TripletMarginMiner,
    BaseMiner,
    MultiSimilarityMiner,
)
from pytorch_metric_learning.distances import LpDistance, BaseDistance
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning.samplers import MPerClassSampler

from deer_face_embed.core.models.base import BaseEmbedderConfig
from deer_face_embed.core.augmentation import AugmentationConfig
from deer_face_embed.core.DeerFaceDataset import DeerFaceDataset

from deer_face_embed.core.evaluation.evaluator import (
    ClosedSetEvaluator,
    MetricsCalculator,
)

# %% Base config for training components


# Base component config with generic create_component method
class ComponentConfig(pdt.BaseModel):
    KIND: str

    def create_component(self, *args, **kwargs):
        """Create component based on config"""
        raise NotImplementedError("Subclasses must implement create_component")


# %% DISTANCES


# Base distance config
class DistanceConfig(ComponentConfig):
    @T.override
    @abc.abstractmethod
    def create_component(self, *args, **kwargs) -> BaseDistance:
        pass


class LpDistanceConfig(DistanceConfig):
    KIND: T.Literal["LpDistance"]
    p: int = 2
    power: int = 1

    @T.override
    def create_component(self, *args, **kwargs):
        return LpDistance(
            p=self.p,
            power=self.power,
            normalize_embeddings=False,
        )


DistanceConfigKind = LpDistanceConfig

# %% MINER


# Base miner config
class MinerConfig(ComponentConfig):
    @T.override
    @abc.abstractmethod
    def create_component(self, *args, **kwargs) -> BaseMiner:
        pass


class TripletMarginMinerConfig(MinerConfig):
    KIND: T.Literal["TripletMarginMiner"]
    margin: float = 0.2
    distance: DistanceConfigKind
    type_of_triplets: str = "all"

    @T.override
    def create_component(self, *args, **kwargs):
        distance = self.distance.create_component()
        return TripletMarginMiner(
            margin=self.margin,
            distance=distance,
            type_of_triplets=self.type_of_triplets,
        )


class MultiSimilarityMinerConfig(MinerConfig):
    KIND: T.Literal["MultiSimilarityMiner"] = "MultiSimilarityMiner"
    epsilon: float = 0.1
    distance: DistanceConfigKind

    @T.override
    def create_component(self, *args, **kwargs):
        distance = self.distance.create_component()
        return MultiSimilarityMiner(epsilon=self.epsilon, distance=distance)


MinerConfigKind = TripletMarginMinerConfig | MultiSimilarityMinerConfig

# %% LOSS


# Base loss config
class LossConfig(ComponentConfig):
    @T.override
    @abc.abstractmethod
    def create_component(self, *args, **kwargs) -> BaseMetricLossFunction:
        pass


class LpRegularizerConfig(pdt.BaseModel):
    """Configuration for LpRegularizer on embeddings."""

    p: int = 2
    power: int = 2


class TripletLossConfig(LossConfig):
    KIND: T.Literal["TripletLoss"]
    margin: float = 0.2
    swap: bool = False
    smooth_loss: bool = False
    embedding_regularizer: LpRegularizerConfig | None = None

    @T.override
    def create_component(self, *args, **kwargs):
        reg = None
        if self.embedding_regularizer:
            reg = LpRegularizer(
                p=self.embedding_regularizer.p,
                power=self.embedding_regularizer.power,
            )
        return TripletMarginLoss(
            margin=self.margin,
            swap=self.swap,
            smooth_loss=self.smooth_loss,
            embedding_regularizer=reg,
        )


LossConfigConfigKind = TripletLossConfig

# %% OPTIMIZER


# Base optimizer config
class OptimizerConfig(ComponentConfig):
    @T.override
    @abc.abstractmethod
    def create_component(self, *args, **kwargs) -> optim.Optimizer:
        pass


class AdamConfig(OptimizerConfig):
    KIND: T.Literal["Adam"]
    learning_rate: float = 0.001

    @T.override
    def create_component(self, *args, **kwargs):
        return optim.Adam(kwargs.get("parameters", []), lr=self.learning_rate)


class AdamWConfig(OptimizerConfig):
    """AdamW optimizer with differential learning rates for backbone vs head."""

    KIND: T.Literal["AdamW"]
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    backbone_lr_factor: float = 0.1

    @T.override
    def create_component(self, *args, **kwargs):
        model = kwargs.get("model")
        if (
            model is not None
            and hasattr(model, "feature_extractor")
            and hasattr(model, "embedding_head")
        ):
            param_groups = [
                {
                    "params": list(model.feature_extractor.parameters()),
                    "lr": self.learning_rate * self.backbone_lr_factor,
                },
                {
                    "params": list(model.embedding_head.parameters()),
                    "lr": self.learning_rate,
                },
            ]
            return optim.AdamW(
                param_groups,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        return optim.AdamW(
            kwargs.get("parameters", []),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


OptimizerConfigKind = AdamConfig | AdamWConfig

# %% EARLY STOPPING


class EarlyStoppingConfig(pdt.BaseModel):
    """Early stopping configuration.

    Monitors a metric and stops training if no improvement is observed
    for `patience` epochs after an initial grace period.
    """

    patience: int = 10
    min_delta: float = 0.001
    monitor: str = "val_loss"
    mode: str = "min"
    start_from_epoch: int = 0
    restore_best_weights: bool = False


# %% LR SCHEDULER


class LRSchedulerConfig(ComponentConfig):
    @T.override
    @abc.abstractmethod
    def create_component(self, *args, **kwargs) -> optim.lr_scheduler.LRScheduler:
        pass


class CosineAnnealingLRConfig(LRSchedulerConfig):
    KIND: T.Literal["CosineAnnealingLR"] = "CosineAnnealingLR"
    T_max: int = 50
    eta_min: float = 1e-6

    @T.override
    def create_component(self, *args, **kwargs):
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer=kwargs["optimizer"],
            T_max=self.T_max,
            eta_min=self.eta_min,
        )


class StepLRConfig(LRSchedulerConfig):
    KIND: T.Literal["StepLR"] = "StepLR"
    step_size: int = 30
    gamma: float = 0.1

    @T.override
    def create_component(self, *args, **kwargs):
        return optim.lr_scheduler.StepLR(
            optimizer=kwargs["optimizer"],
            step_size=self.step_size,
            gamma=self.gamma,
        )


class ReduceLROnPlateauConfig(LRSchedulerConfig):
    KIND: T.Literal["ReduceLROnPlateau"] = "ReduceLROnPlateau"
    patience: int = 10
    factor: float = 0.1
    min_lr: float = 1e-6

    @T.override
    def create_component(self, *args, **kwargs):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=kwargs["optimizer"],
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
        )


class CosineAnnealingWarmRestartsConfig(LRSchedulerConfig):
    KIND: T.Literal["CosineAnnealingWarmRestarts"] = "CosineAnnealingWarmRestarts"
    T_0: int = 30
    T_mult: int = 2
    eta_min: float = 1e-7

    @T.override
    def create_component(self, *args, **kwargs):
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=kwargs["optimizer"],
            T_0=self.T_0,
            T_mult=self.T_mult,
            eta_min=self.eta_min,
        )


LRSchedulerConfigKind = (
    CosineAnnealingLRConfig
    | StepLRConfig
    | ReduceLROnPlateauConfig
    | CosineAnnealingWarmRestartsConfig
)

# %% TRAINING


class WarmupConfig(pdt.BaseModel):
    """Linear warmup configuration."""

    warmup_epochs: int = 15
    warmup_start_lr: float = 1e-7


class MPerClassSamplerConfig(pdt.BaseModel):
    """MPerClassSampler configuration for balanced batch construction."""

    m: int = 4


# Training configuration
class TrainingConfig(pdt.BaseModel):
    # required parameters
    num_epochs: int

    # training components
    miner: MinerConfigKind = pdt.Field(..., discriminator="KIND")
    loss: LossConfigConfigKind = pdt.Field(..., discriminator="KIND")
    optimizer: OptimizerConfigKind = pdt.Field(..., discriminator="KIND")
    augmentation_config: AugmentationConfig = AugmentationConfig()

    # optional params
    validation_split: float = 0.2
    split_seed: int = 123
    dropout_rate: float = 0.3

    # optional training mechanisms (disabled by default)
    early_stopping: EarlyStoppingConfig | None = None
    lr_scheduler: LRSchedulerConfigKind | None = pdt.Field(
        default=None, discriminator="KIND"
    )
    warmup: WarmupConfig | None = None
    sampler: MPerClassSamplerConfig | None = None
    gradient_clip_val: float | None = None


class Trainer:
    def __init__(
        self,
        train_config: TrainingConfig,
        model_config: BaseEmbedderConfig,
        train_indices: Sequence,
        val_indices: Sequence,
        dataset_metadata: pd.DataFrame,
        images_base_path: str,
    ):
        """
        Initialize the Trainer with model, configuration, and dataset
        """
        self._logger = logging.getLogger(__name__)
        self.device = self._init_device()
        self.train_config: TrainingConfig = train_config

        self.model_config = model_config
        self.model = self.model_config.load_model(self.device)

        layer_groups = self.model.get_layer_groups()
        for i, params_list in sorted(layer_groups.items()):
            n_params = sum(p.numel() for p in params_list)
            trainable = sum(p.numel() for p in params_list if p.requires_grad)
            self._logger.info(
                f"group {i}: {n_params / 1e6:.1f}M params "
                f"({trainable / 1e6:.1f}M trainable)"
            )

        # Setup components based on configuration
        self.loss_func = self.train_config.loss.create_component()
        self.miner = self.train_config.miner.create_component()
        self.optimizer = self.train_config.optimizer.create_component(
            parameters=self.model.parameters(),
            model=self.model,
        )
        self.scheduler = (
            self.train_config.lr_scheduler.create_component(optimizer=self.optimizer)
            if self.train_config.lr_scheduler
            else None
        )

        # Set up data
        self._prepare_dataset(
            dataset_metadata, images_base_path, train_indices, val_indices
        )
        self._prepare_dataloader()

    def _init_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._logger.info(f"Using device: {device}")
        return device

    def _get_transforms(self):
        self._logger.info("Setting up data transforms...")
        inference_transform = self.model.generate_transform(no_cropping=True)
        augmentations = self.train_config.augmentation_config.generate_aug_transforms()
        train_transform = Compose([augmentations, inference_transform])
        self._logger.info("Transforms set.")
        return train_transform, inference_transform

    def _prepare_dataset(
        self, dataset_metadata, images_base_path, train_indices, val_indices
    ):
        train_transform, inference_transform = self._get_transforms()
        train_metadata = dataset_metadata.loc[train_indices].reset_index(drop=True)
        val_metadata = dataset_metadata.loc[val_indices].reset_index(drop=True)

        # Shared label encoding so train/val use the same identity → int mapping
        all_identities = sorted(dataset_metadata["identity"].unique())
        label_encoding = {identity: idx for idx, identity in enumerate(all_identities)}

        self.train_dataset = DeerFaceDataset(
            metadata=train_metadata,
            root=images_base_path,
            transform=train_transform,
            label_encoding=label_encoding,
        )
        self.train_eval_dataset = DeerFaceDataset(
            metadata=train_metadata,
            root=images_base_path,
            transform=inference_transform,
            label_encoding=label_encoding,
        )
        self.val_dataset = DeerFaceDataset(
            metadata=val_metadata,
            root=images_base_path,
            transform=inference_transform,
            label_encoding=label_encoding,
        )

    def _prepare_dataloader(self):
        sampler = None
        shuffle = True
        if self.train_config.sampler is not None:
            sampler = MPerClassSampler(
                labels=self.train_dataset.labels.tolist(),
                m=self.train_config.sampler.m,
                length_before_new_iter=len(self.train_dataset),
            )
            shuffle = False  # sampler and shuffle are mutually exclusive

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
        )

        self.train_eval_loader = DataLoader(
            self.train_eval_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.model_config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    def _train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        num_batches = len(self.train_loader)

        for data, labels in tqdm(self.train_loader, total=num_batches, desc="Training"):
            data, labels = data.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            embeddings = self.model(data)
            mined_triplets = self.miner(embeddings, labels)
            loss = self.loss_func(embeddings, labels, mined_triplets)
            loss.backward()

            if self.train_config.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.train_config.gradient_clip_val,
                )

            self.optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / num_batches
        return average_loss

    def run_train_loop(self) -> None:
        num_epochs = self.train_config.num_epochs
        early_stop_cfg = self.train_config.early_stopping
        warmup_cfg = self.train_config.warmup
        freeze_cfg = self.model_config.backbone_freeze
        transition_epoch = freeze_cfg.transition_epoch if freeze_cfg else 0

        # Best model tracking (always active, independent of early stopping)
        best_monitor = "precision_at_1.val"
        best_mode = "max"
        best_metric_value: float | None = None
        best_model_state: dict | None = None

        # Early stopping state
        patience_counter = 0

        if early_stop_cfg:
            # Early stopping reuses the same monitor/mode for consistency
            best_monitor = early_stop_cfg.monitor
            best_mode = early_stop_cfg.mode
            self._logger.info(
                f"Early stopping enabled: monitor={early_stop_cfg.monitor}, "
                f"patience={early_stop_cfg.patience}, mode={early_stop_cfg.mode}, "
                f"start_from_epoch={early_stop_cfg.start_from_epoch}"
            )
        self._logger.info(
            f"Best model tracking: monitor={best_monitor}, mode={best_mode}"
        )
        if self.scheduler:
            self._logger.info(
                f"LR scheduler enabled: {self.train_config.lr_scheduler.KIND}"  # type: ignore
            )
        if warmup_cfg:
            self._logger.info(
                f"Warmup enabled: {warmup_cfg.warmup_epochs} epochs, "
                f"start_lr={warmup_cfg.warmup_start_lr}"
            )
        if transition_epoch > 0:
            self._logger.info(f"Freeze phase transition after epoch {transition_epoch}")

        # Store target LRs for warmup interpolation
        target_lrs: list[float] = []
        if warmup_cfg:
            target_lrs = [pg["lr"] for pg in self.optimizer.param_groups]

        for epoch in range(num_epochs):
            self._logger.info(f"EPOCH {epoch + 1}/{num_epochs}")

            # --- Warmup: override LR for warmup epochs ---
            if warmup_cfg and epoch < warmup_cfg.warmup_epochs:
                warmup_fraction = (epoch + 1) / warmup_cfg.warmup_epochs
                for i, pg in enumerate(self.optimizer.param_groups):
                    pg["lr"] = (
                        warmup_cfg.warmup_start_lr
                        + (target_lrs[i] - warmup_cfg.warmup_start_lr) * warmup_fraction
                    )
                current_lr = self.optimizer.param_groups[0]["lr"]
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

            # --- Freeze phase transition ---
            if transition_epoch > 0 and epoch == transition_epoch:
                final_phase = freeze_cfg.final if freeze_cfg else None
                self._logger.info(
                    f"Switching to final freeze phase at epoch {epoch + 1}"
                )
                self.model.apply_freeze_phase(final_phase)
                # Recreate optimizer and scheduler for newly unfrozen params
                self.optimizer = self.train_config.optimizer.create_component(
                    parameters=self.model.parameters(),
                    model=self.model,
                )
                if self.train_config.lr_scheduler:
                    self.scheduler = self.train_config.lr_scheduler.create_component(
                        optimizer=self.optimizer
                    )
                # Reset warmup targets for the new optimizer
                if warmup_cfg:
                    target_lrs = [pg["lr"] for pg in self.optimizer.param_groups]

            # --- Train one epoch ---
            train_loss = self._train_one_epoch()
            mlflow.log_metric("loss.train", train_loss, step=epoch)
            self._logger.info(f"Train Loss: {train_loss:.4f}")

            # --- Evaluate ---
            epoch_metrics = self.evaluate_epoch(epoch, num_epochs)

            # --- LR scheduler step (skip during warmup) ---
            if self.scheduler and (
                warmup_cfg is None or epoch >= warmup_cfg.warmup_epochs
            ):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_metrics["val_loss"])
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                mlflow.log_metric("learning_rate", current_lr, step=epoch)

            # --- Best model tracking (always active) ---
            current_value = epoch_metrics.get(best_monitor)
            if current_value is not None:
                is_best = best_metric_value is None or (
                    current_value < best_metric_value
                    if best_mode == "min"
                    else current_value > best_metric_value
                )
                if is_best:
                    best_metric_value = current_value
                    best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    self._logger.info(
                        f"New best {best_monitor}={current_value:.4f} "
                        f"at epoch {epoch + 1}, saving checkpoint"
                    )
                    mlflow.log_metric("best_epoch", epoch + 1)
                    mlflow.log_metric(f"best_{best_monitor}", current_value, epoch)

            # --- Early stopping check (uses min_delta for patience) ---
            if early_stop_cfg and epoch >= early_stop_cfg.start_from_epoch:
                es_value = epoch_metrics.get(
                    early_stop_cfg.monitor, epoch_metrics["val_loss"]
                )
                # Check if improvement exceeds min_delta threshold
                if not hasattr(self, "_es_best_value"):
                    self._es_best_value = es_value
                    patience_counter = 0
                else:
                    es_improved = (
                        es_value < self._es_best_value - early_stop_cfg.min_delta
                        if early_stop_cfg.mode == "min"
                        else es_value > self._es_best_value + early_stop_cfg.min_delta
                    )
                    if es_improved:
                        self._es_best_value = es_value
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        self._logger.info(
                            f"Early stopping: no improvement for "
                            f"{patience_counter}/{early_stop_cfg.patience} epochs"
                        )
                        if patience_counter >= early_stop_cfg.patience:
                            self._logger.info(
                                f"Early stopping triggered at epoch {epoch + 1}"
                            )
                            mlflow.log_metric("early_stopping.epoch", epoch + 1)
                            if (
                                early_stop_cfg.restore_best_weights
                                and best_model_state is not None
                            ):
                                self._logger.info("Restoring best model weights")
                                self.model.load_state_dict(best_model_state)
                            break

        # After training completes, restore best weights if available
        if best_model_state is not None:
            should_restore = (
                early_stop_cfg is not None and early_stop_cfg.restore_best_weights
            ) or early_stop_cfg is None
            if should_restore:
                self._logger.info("Training complete. Restoring best model weights.")
                self.model.load_state_dict(best_model_state)

        # Log best model to MLflow once after training (not per-epoch)
        if best_model_state is not None:
            mlflow.pytorch.log_model(
                pytorch_model=self.model,
                artifact_path="best_model",
            )

    def evaluate_epoch(self, epoch: int, num_epochs: int) -> dict[str, float]:
        train_embeddings, train_labels = self.model.infer_embeddings(
            self.train_eval_loader
        )
        val_embeddings, val_labels = self.model.infer_embeddings(self.val_loader)

        # Calc validation loss
        val_loss = self.loss_func(val_embeddings, val_labels)
        mlflow.log_metric("loss.val", val_loss, step=epoch)
        self._logger.info(f"Val Loss: {val_loss:.4f}")

        metrics: dict[str, float] = {"val_loss": float(val_loss)}

        # Accuracies on seen images
        train_metrics = MetricsCalculator.compute_metrics(
            query_embeddings=train_embeddings,
            query_labels=train_labels,
            reference_embeddings=train_embeddings,
            reference_labels=train_labels,
        )
        if "precision_at_1" in train_metrics:
            self._logger.info(
                f"Train precision_at_1: {train_metrics['precision_at_1']:.4f}"
            )
        for k, v in train_metrics.items():
            mlflow.log_metric(f"{k}.train", v, step=epoch)

        # Accuracies on unseen images, but known individuals
        val_metrics = MetricsCalculator.compute_metrics(
            query_embeddings=val_embeddings,
            query_labels=val_labels,
            reference_embeddings=train_embeddings,
            reference_labels=train_labels,
        )
        if "precision_at_1" in val_metrics:
            self._logger.info(
                f"Val precision_at_1: {val_metrics['precision_at_1']:.4f}"
            )
        for k, v in val_metrics.items():
            mlflow.log_metric(f"{k}.val", v, step=epoch)
            metrics[f"{k}.val"] = v

        closed_set_eval = ClosedSetEvaluator(val_embeddings, val_labels)
        (fpr, tpr), roc_auc = closed_set_eval.calc_roc()
        mlflow.log_metric("roc.val", roc_auc, step=epoch)
        metrics["roc.val"] = roc_auc

        # Free cached GPU memory after evaluation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return metrics
