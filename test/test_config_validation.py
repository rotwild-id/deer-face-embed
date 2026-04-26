"""Tests for pydantic config discriminated unions (embedder, loss, miner)."""

import pytest
import pydantic

from deer_face_embed.core.models.base import (
    BackboneFreezeConfig,
    BaseEmbedderConfig,
    FreezePhaseConfig,
)
from deer_face_embed.core.models.CNNBasedModels import ResNetEmbedderConfig
from deer_face_embed.core.models.VitBasedModels import (
    VitBaseFaceEmbedderConfig,
    VitDinoFaceEmbedderConfig,
    SwinFaceEmbedderConfig,
)
from deer_face_embed.core.trainer import (
    TripletLossConfig,
    TripletMarginMinerConfig,
    MultiSimilarityMinerConfig,
    LpDistanceConfig,
    AdamConfig,
    AdamWConfig,
)


# ---------------------------------------------------------------------------
# Helper to make a minimal LpDistance config (required by miners)
# ---------------------------------------------------------------------------


def _distance():
    return LpDistanceConfig(KIND="LpDistance")


# ---------------------------------------------------------------------------
# EmbedderConfig validation
# ---------------------------------------------------------------------------


class TestEmbedderConfigValidation:
    def test_resnet_config_valid(self):
        cfg = ResNetEmbedderConfig(
            pretrained=False,
            embedding_size=128,
        )
        assert cfg.KIND == "ResNet"
        assert cfg.pretrained is False
        assert cfg.embedding_size == 128

    def test_vitbase_config_valid(self):
        cfg = VitBaseFaceEmbedderConfig(
            pretrained=False,
            embedding_size=256,
        )
        assert cfg.KIND == "VitBase"

    def test_swin_config_valid(self):
        cfg = SwinFaceEmbedderConfig(
            pretrained=False,
            embedding_size=64,
        )
        assert cfg.KIND == "Swin"

    def test_invalid_extra_field_raises(self):
        """extra='forbid' means unknown fields should raise a ValidationError."""
        with pytest.raises(pydantic.ValidationError):
            ResNetEmbedderConfig(
                pretrained=False,
                embedding_size=128,
                non_existent_field=True,  # type: ignore[call-arg]
            )

    def test_freeze_requires_pretrained(self):
        """backbone_freeze on a non-pretrained model should raise ValidationError."""
        freeze_cfg = BackboneFreezeConfig(
            initial=FreezePhaseConfig(trainable_groups=[]),
            transition_epoch=0,
            final=None,
        )
        with pytest.raises(pydantic.ValidationError, match="pretrained"):
            ResNetEmbedderConfig(
                pretrained=False,
                embedding_size=128,
                backbone_freeze=freeze_cfg,
            )

    def test_default_batch_size(self):
        cfg = ResNetEmbedderConfig(pretrained=False, embedding_size=64)
        assert cfg.batch_size == 32

    def test_default_intermediate_size(self):
        cfg = ResNetEmbedderConfig(pretrained=False, embedding_size=64)
        assert cfg.intermediate_size == 1024

    def test_default_dropout_rate(self):
        cfg = ResNetEmbedderConfig(pretrained=False, embedding_size=64)
        assert cfg.dropout_rate == 0.3


# ---------------------------------------------------------------------------
# LossConfig validation
# ---------------------------------------------------------------------------


class TestLossConfigValidation:
    def test_triplet_loss_valid(self):
        cfg = TripletLossConfig(KIND="TripletLoss", margin=0.2)
        assert cfg.KIND == "TripletLoss"
        assert cfg.margin == 0.2

    def test_triplet_loss_defaults(self):
        cfg = TripletLossConfig(KIND="TripletLoss")
        assert cfg.swap is False
        assert cfg.smooth_loss is False
        assert cfg.embedding_regularizer is None


# ---------------------------------------------------------------------------
# MinerConfig validation
# ---------------------------------------------------------------------------


class TestMinerConfigValidation:
    def test_triplet_margin_miner_valid(self):
        cfg = TripletMarginMinerConfig(
            KIND="TripletMarginMiner",
            margin=0.2,
            distance=_distance(),
        )
        assert cfg.KIND == "TripletMarginMiner"

    def test_multi_similarity_miner_valid(self):
        cfg = MultiSimilarityMinerConfig(
            KIND="MultiSimilarityMiner",
            epsilon=0.1,
            distance=_distance(),
        )
        assert cfg.KIND == "MultiSimilarityMiner"

    def test_triplet_miner_default_type_of_triplets(self):
        cfg = TripletMarginMinerConfig(
            KIND="TripletMarginMiner",
            distance=_distance(),
        )
        assert cfg.type_of_triplets == "all"


# ---------------------------------------------------------------------------
# OptimizerConfig validation
# ---------------------------------------------------------------------------


class TestOptimizerConfigValidation:
    def test_adam_valid(self):
        cfg = AdamConfig(KIND="Adam", learning_rate=0.001)
        assert cfg.KIND == "Adam"

    def test_adamw_valid(self):
        cfg = AdamWConfig(
            KIND="AdamW",
            learning_rate=0.001,
            weight_decay=0.01,
            backbone_lr_factor=0.1,
        )
        assert cfg.KIND == "AdamW"
        assert cfg.weight_decay == 0.01

    def test_adam_default_lr(self):
        cfg = AdamConfig(KIND="Adam")
        assert cfg.learning_rate == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# FreezePhaseConfig
# ---------------------------------------------------------------------------


class TestFreezeConfigValidation:
    def test_freeze_phase_defaults(self):
        cfg = FreezePhaseConfig()
        assert cfg.trainable_groups == []

    def test_backbone_freeze_defaults(self):
        cfg = BackboneFreezeConfig()
        assert cfg.transition_epoch == 0
        assert cfg.final is None

    def test_freeze_phase_with_groups(self):
        cfg = FreezePhaseConfig(trainable_groups=[0, 1, 2])
        assert cfg.trainable_groups == [0, 1, 2]
