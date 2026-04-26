"""Tests for model instantiation, forward pass, and L2 normalisation.

All tests use CPU only (pretrained=False to avoid weight downloads).
"""

import pytest
import torch
import torch.nn as nn

from deer_face_embed.core.models.CNNBasedModels import ResNetEmbedderConfig
from deer_face_embed.core.models.VitBasedModels import VitBaseFaceEmbedderConfig
from deer_face_embed.core.models.base import (
    BackboneFreezeConfig,
    FreezePhaseConfig,
)

DEVICE = torch.device("cpu")
EMBEDDING_SIZE = 64
BATCH = 2
INPUT = (BATCH, 3, 224, 224)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resnet_config(**kwargs) -> ResNetEmbedderConfig:
    defaults = dict(
        pretrained=False,
        embedding_size=EMBEDDING_SIZE,
    )
    defaults.update(kwargs)
    return ResNetEmbedderConfig(**defaults)


def _vitbase_config(**kwargs) -> VitBaseFaceEmbedderConfig:
    defaults = dict(
        pretrained=False,
        embedding_size=EMBEDDING_SIZE,
    )
    defaults.update(kwargs)
    return VitBaseFaceEmbedderConfig(**defaults)


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestModelInstantiation:
    def test_resnet_instantiates(self):
        model = _resnet_config().load_model(device=DEVICE)
        assert model is not None
        assert isinstance(model, nn.Module)

    def test_vitbase_instantiates(self):
        model = _vitbase_config().load_model(device=DEVICE)
        assert model is not None
        assert isinstance(model, nn.Module)


# ---------------------------------------------------------------------------
# Forward pass shape
# ---------------------------------------------------------------------------


class TestForwardPassShape:
    def test_resnet_output_shape(self):
        model = _resnet_config().load_model(device=DEVICE)
        model.eval()
        x = torch.rand(*INPUT, device=DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, EMBEDDING_SIZE), (
            f"Expected ({BATCH}, {EMBEDDING_SIZE}), got {out.shape}"
        )

    def test_vitbase_output_shape(self):
        model = _vitbase_config().load_model(device=DEVICE)
        model.eval()
        x = torch.rand(*INPUT, device=DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, EMBEDDING_SIZE), (
            f"Expected ({BATCH}, {EMBEDDING_SIZE}), got {out.shape}"
        )


# ---------------------------------------------------------------------------
# L2 normalisation
# ---------------------------------------------------------------------------


class TestL2Normalisation:
    def _check_unit_norm(self, model):
        model.eval()
        x = torch.rand(*INPUT, device=DEVICE)
        with torch.no_grad():
            out = model(x)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(BATCH, device=DEVICE), atol=1e-5), (
            f"Output vectors should have unit L2 norm; got norms={norms}"
        )

    def test_resnet_l2_norm(self):
        self._check_unit_norm(_resnet_config().load_model(device=DEVICE))

    def test_vitbase_l2_norm(self):
        self._check_unit_norm(_vitbase_config().load_model(device=DEVICE))


# ---------------------------------------------------------------------------
# Backbone freezing
# ---------------------------------------------------------------------------


class TestBackboneFreezing:
    def test_freeze_all_backbone_params(self):
        """When trainable_groups=[], all backbone params should be frozen."""
        freeze_cfg = BackboneFreezeConfig(
            initial=FreezePhaseConfig(trainable_groups=[]),
            transition_epoch=0,
            final=None,
        )
        # backbone_freeze requires pretrained=True; we skip if model load fails
        # because pretrained weights are not available in CI.
        # Instead, test apply_freeze_phase directly on a pretrained=False model.
        model = _resnet_config().load_model(device=DEVICE)
        # Freeze all backbone params via the phase config
        phase = FreezePhaseConfig(trainable_groups=[])
        model.apply_freeze_phase(phase)
        for param in model.feature_extractor.parameters():
            assert not param.requires_grad, (
                "All backbone params should be frozen when trainable_groups=[]"
            )

    def test_unfreeze_all_backbone_params(self):
        """apply_freeze_phase(None) should unfreeze all backbone parameters."""
        model = _resnet_config().load_model(device=DEVICE)
        # First freeze everything
        model.apply_freeze_phase(FreezePhaseConfig(trainable_groups=[]))
        # Then unfreeze
        model.apply_freeze_phase(None)
        for param in model.feature_extractor.parameters():
            assert param.requires_grad, (
                "All backbone params should be trainable after apply_freeze_phase(None)"
            )

    def test_partial_unfreeze_layer_group(self):
        """Specifying trainable_groups=[3] unfreezes only group 3 (layer4)."""
        model = _resnet_config().load_model(device=DEVICE)
        phase = FreezePhaseConfig(trainable_groups=[3])
        model.apply_freeze_phase(phase)

        # layer4 params (group 3) must be trainable
        layer4_params = set(id(p) for p in model.feature_extractor.layer4.parameters())
        for param in model.feature_extractor.parameters():
            if id(param) in layer4_params:
                assert param.requires_grad, "layer4 params should be trainable"
            else:
                assert not param.requires_grad, (
                    "Params outside layer4 should be frozen"
                )

    def test_get_layer_groups_returns_four_groups(self):
        """ResNet should expose exactly 4 layer groups (0–3)."""
        model = _resnet_config().load_model(device=DEVICE)
        groups = model.get_layer_groups()
        assert set(groups.keys()) == {0, 1, 2, 3}
        for idx, params in groups.items():
            assert len(params) > 0, f"Group {idx} should have at least one parameter"

    def test_invalid_group_raises(self):
        """apply_freeze_phase with an out-of-range group index should raise ValueError."""
        model = _resnet_config().load_model(device=DEVICE)
        with pytest.raises(ValueError, match="Group"):
            model.apply_freeze_phase(FreezePhaseConfig(trainable_groups=[99]))
