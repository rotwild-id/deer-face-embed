"""Tests for augmentation config classes and transform generation."""

import torch
import pytest
from PIL import Image
from torchvision.transforms.v2 import Compose, Identity, Transform

from deer_face_embed.core.augmentation import (
    AugmentationConfig,
    ColorJitterAug,
    RandomAdjustSharpnessAug,
    RandomErasingAug,
    RandomGaussianBlurAug,
    RandomGaussianNoiseAug,
    RandomGrayscaleAug,
    RandomHorizontalFlipAug,
    RandomPhotometricDistortAug,
    RandomResizedCropAug,
    RandomRotationAug,
    ResizeAug,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_pil_image(size=(64, 64)):
    import numpy as np

    arr = (torch.rand(3, *size) * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Individual augmentation configs build valid transforms
# ---------------------------------------------------------------------------


class TestAugmentationBuild:
    """Each augmentation config's get_transform() should return a callable Transform."""

    def _assert_transform_works(self, aug):
        t = aug.get_transform()
        assert callable(t), f"{type(aug).__name__}.get_transform() must be callable"
        # Should be able to apply it to a PIL image without error
        img = _make_pil_image()
        _ = t(img)

    def test_random_horizontal_flip(self):
        self._assert_transform_works(RandomHorizontalFlipAug(p=0.5))

    def test_random_grayscale(self):
        self._assert_transform_works(RandomGrayscaleAug(p=0.5))

    def test_random_photometric_distort(self):
        self._assert_transform_works(RandomPhotometricDistortAug())

    def test_random_adjust_sharpness(self):
        self._assert_transform_works(RandomAdjustSharpnessAug())

    def test_resize(self):
        self._assert_transform_works(ResizeAug(size=64))

    def test_random_gaussian_blur(self):
        self._assert_transform_works(RandomGaussianBlurAug())

    def test_random_rotation(self):
        self._assert_transform_works(RandomRotationAug())

    def test_random_erasing(self):
        # RandomErasing expects a tensor, not PIL
        aug = RandomErasingAug()
        t = aug.get_transform()
        assert callable(t)
        tensor = torch.rand(3, 64, 64)
        _ = t(tensor)

    def test_random_resized_crop(self):
        self._assert_transform_works(RandomResizedCropAug(size=64))

    def test_color_jitter(self):
        self._assert_transform_works(ColorJitterAug())

    def test_random_gaussian_noise(self):
        self._assert_transform_works(RandomGaussianNoiseAug())


# ---------------------------------------------------------------------------
# AugmentationConfig pipeline
# ---------------------------------------------------------------------------


class TestAugmentationConfig:
    def test_default_config_produces_pipeline(self):
        """Default AugmentationConfig should produce a non-empty Compose pipeline."""
        cfg = AugmentationConfig()
        transform = cfg.generate_aug_transforms()
        # Should be a Compose (multiple augmentations)
        assert isinstance(transform, Compose), (
            "Default config should produce a Compose pipeline"
        )
        assert len(cfg.augmentations) > 0, "Default config must have augmentations"

    def test_empty_augmentation_list_produces_identity(self):
        """An empty augmentation list should produce an Identity transform."""
        cfg = AugmentationConfig(augmentations=[])
        transform = cfg.generate_aug_transforms()
        assert isinstance(transform, Identity), (
            "Empty augmentation list should produce Identity transform"
        )

    def test_single_augmentation_produces_compose(self):
        """A single augmentation should still be wrapped in Compose."""
        cfg = AugmentationConfig(augmentations=[RandomHorizontalFlipAug(p=0.5)])
        transform = cfg.generate_aug_transforms()
        # Compose is returned for any non-empty list
        assert isinstance(transform, Compose)

    def test_pipeline_is_applicable_to_pil(self):
        """The default pipeline should be applicable to a PIL Image."""
        cfg = AugmentationConfig()
        transform = cfg.generate_aug_transforms()
        img = _make_pil_image()
        # Should not raise
        _ = transform(img)

    def test_augmentations_discriminated_union_kind(self):
        """Configs should be recognised by their KIND discriminator."""
        cfg = AugmentationConfig(
            augmentations=[
                {"KIND": "RandomHorizontalFlip", "p": 0.3},
                {"KIND": "RandomGrayscale", "p": 0.1},
            ]
        )
        assert len(cfg.augmentations) == 2
        assert isinstance(cfg.augmentations[0], RandomHorizontalFlipAug)
        assert isinstance(cfg.augmentations[1], RandomGrayscaleAug)
