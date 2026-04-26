import abc
import typing as T
import pydantic as pdt

import torch
from PIL import Image
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2 import (
    Compose,
    RandomPhotometricDistort,
    RandomHorizontalFlip,
    RandomGrayscale,
    RandomAdjustSharpness,
    Resize,
    GaussianBlur,
    RandomRotation,
    RandomErasing,
    RandomResizedCrop,
    ColorJitter,
    Transform,
    Identity,
)


# Base class for all augmentations
class AugmentationBase(pdt.BaseModel, abc.ABC):
    """Base class for all augmentation configurations."""

    pass

    @abc.abstractmethod
    def get_transform(self) -> Transform:
        pass


class RandomPhotometricDistortAug(AugmentationBase):
    """Configuration for RandomPhotometricDistort augmentation."""

    KIND: T.Literal["RandomPhotometricDistort"] = "RandomPhotometricDistort"

    # Default values equal to pytorch default value
    brightness: tuple[float, float] = (0.875, 1.125)
    contrast: tuple[float, float] = (0.5, 1.5)
    saturation: tuple[float, float] = (0.5, 1.5)
    hue: tuple[float, float] = (-0.01, 0.01)
    p: float = 0.5

    @T.override
    def get_transform(self) -> Transform:
        return RandomPhotometricDistort(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
            p=self.p,
        )


class RandomHorizontalFlipAug(AugmentationBase):
    """Configuration for RandomHorizontalFlip augmentation."""

    KIND: T.Literal["RandomHorizontalFlip"] = "RandomHorizontalFlip"
    p: float = 0.5

    @T.override
    def get_transform(self) -> Transform:
        return RandomHorizontalFlip(p=self.p)


class RandomGrayscaleAug(AugmentationBase):
    """Configuration for RandomGrayscale augmentation."""

    KIND: T.Literal["RandomGrayscale"] = "RandomGrayscale"
    p: float = 0.5

    @T.override
    def get_transform(self) -> Transform:
        return RandomGrayscale(p=self.p)


class RandomAdjustSharpnessAug(AugmentationBase):
    """Configuration for RandomAdjustSharpness augmentation.
    More information: https://pytorch.org/vision/0.15/generated/torchvision.transforms.v2.RandomAdjustSharpness.html
    """

    KIND: T.Literal["RandomAdjustSharpness"] = "RandomAdjustSharpness"
    sharpness_factor: float = 2
    p: float = 0.5

    @T.override
    def get_transform(self) -> Transform:
        return RandomAdjustSharpness(sharpness_factor=self.sharpness_factor, p=self.p)


class ResizeAug(AugmentationBase):
    """Configuration for Resize augmentation."""

    KIND: T.Literal["Resize"] = "Resize"
    size: T.Union[int, list[int]]

    @T.override
    def get_transform(self) -> Transform:
        size = self.size if isinstance(self.size, int) else tuple(self.size)
        return Resize(size=size)


class RandomGaussianBlurAug(AugmentationBase):
    """Configuration for GaussianBlur augmentation with random sigma.

    Simulates camera motion blur and focus issues. Safe with black-padded
    mask chips since blur preserves dark regions.
    """

    KIND: T.Literal["RandomGaussianBlur"] = "RandomGaussianBlur"
    kernel_size: T.List[int] = [3, 7]
    sigma: tuple[float, float] = (0.1, 2.0)
    p: float = 0.3

    @T.override
    def get_transform(self) -> Transform:
        return GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)


class RandomRotationAug(AugmentationBase):
    """Configuration for RandomRotation augmentation.

    Handles head tilt and camera angle variation. Fills rotated corners with
    black (fill=0) to remain consistent with mask-based black padding.
    """

    KIND: T.Literal["RandomRotation"] = "RandomRotation"
    degrees: T.List[float] = [-15, 15]
    fill: int = 0
    p: float = 0.5

    @T.override
    def get_transform(self) -> Transform:
        return RandomRotation(degrees=self.degrees, fill=self.fill)


class RandomErasingAug(AugmentationBase):
    """Configuration for RandomErasing augmentation.

    Improves robustness to occlusions by erasing random rectangular regions.
    Filling with black (value=0) matches mask-based chip padding style.
    Proven effective in metric learning / person re-identification tasks.
    """

    KIND: T.Literal["RandomErasing"] = "RandomErasing"
    p: float = 0.3
    scale: tuple[float, float] = (0.02, 0.15)
    ratio: tuple[float, float] = (0.3, 3.3)
    value: int = 0

    @T.override
    def get_transform(self) -> Transform:
        return RandomErasing(
            p=self.p,
            scale=self.scale,
            ratio=self.ratio,
            value=self.value,
        )


class RandomResizedCropAug(AugmentationBase):
    """Configuration for RandomResizedCrop augmentation.

    More aggressive cropping than default improves robustness to scale variation.
    """

    KIND: T.Literal["RandomResizedCrop"] = "RandomResizedCrop"
    size: T.Union[int, list[int]] = 224
    scale: tuple[float, float] = (0.8, 1.0)
    ratio: tuple[float, float] = (0.75, 1.333)

    @T.override
    def get_transform(self) -> Transform:
        size = self.size if isinstance(self.size, int) else tuple(self.size)
        return RandomResizedCrop(
            size=size,
            scale=self.scale,
            ratio=self.ratio,
        )


class ColorJitterAug(AugmentationBase):
    """Configuration for ColorJitter augmentation."""

    KIND: T.Literal["ColorJitter"] = "ColorJitter"
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05

    @T.override
    def get_transform(self) -> Transform:
        return ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )


class _GaussianNoiseTransform(torch.nn.Module):
    """Adds Gaussian noise to a tensor image with a given probability."""

    def __init__(self, mean: float = 0.0, std: float = 0.05, p: float = 0.3):
        super().__init__()
        self.mean = mean
        self.std = std
        self.p = p

    def forward(self, img: torch.Tensor | Image.Image) -> torch.Tensor | Image.Image:
        if torch.rand(1).item() < self.p:
            if isinstance(img, Image.Image):
                tensor = F.to_image(img).float()
                noise = torch.randn_like(tensor) * self.std + self.mean
                result = (tensor + noise).clamp(0, 255).to(torch.uint8)
                return F.to_pil_image(result)
            tensor = img.float()
            noise = torch.randn_like(tensor) * self.std + self.mean
            result = (tensor + noise).clamp(0, 255).to(torch.uint8)
            return result.to(img.dtype)
        return img


class RandomGaussianNoiseAug(AugmentationBase):
    """Configuration for Gaussian noise augmentation."""

    KIND: T.Literal["RandomGaussianNoise"] = "RandomGaussianNoise"
    mean: float = 0.0
    std: float = 0.05
    p: float = 0.3

    @T.override
    def get_transform(self) -> Transform:
        return _GaussianNoiseTransform(mean=self.mean, std=self.std, p=self.p)  # type: ignore


# Define the discriminated union using Annotated
AugmentationKind = T.Annotated[
    RandomPhotometricDistortAug
    | RandomHorizontalFlipAug
    | RandomGrayscaleAug
    | RandomAdjustSharpnessAug
    | ResizeAug
    | RandomGaussianBlurAug
    | RandomRotationAug
    | RandomErasingAug
    | RandomResizedCropAug
    | ColorJitterAug
    | RandomGaussianNoiseAug,
    pdt.Field(discriminator="KIND"),
]


class AugmentationConfig(pdt.BaseModel):
    """
    Configuration class for image augmentations.

    This class uses Pydantic discriminated unions to configure different
    augmentation transforms based on a YAML-style configuration.
    """

    augmentations: T.List[AugmentationKind] = pdt.Field(
        default_factory=lambda: AugmentationConfig.default_augmentations()
    )

    def generate_aug_transforms(self) -> Transform:
        """
        Generate augmentation transforms based on the configuration.

        Returns:
            A composition of torchvision transforms.
        """
        if len(self.augmentations) == 0:
            # no augmentation is configured
            return Identity()
        else:
            transforms = [aug.get_transform() for aug in self.augmentations]
            return Compose(transforms)

    @classmethod
    def default_augmentations(cls) -> T.List[AugmentationKind]:
        """
        Default augmentations optimized for SAM3 mask-based face chips with black padding.

        Returns:
            Conservative augmentation pipeline safe for black-padded regions.
        """
        return [
            # Geometric (safe with padding)
            RandomHorizontalFlipAug(p=0.5),
            # Photometric (conservative ranges to avoid padding brightness artefacts)
            RandomPhotometricDistortAug(
                brightness=(0.95, 1.05),
                contrast=(0.85, 1.15),
                saturation=(0.8, 1.2),
                hue=(-0.01, 0.01),
                p=0.5,
            ),
            # Grayscale (increased probability for better regularisation)
            RandomGrayscaleAug(p=0.2),
            # Wildlife-specific: blur simulates camera motion/focus issues
            RandomGaussianBlurAug(
                kernel_size=[3, 7],
                sigma=(0.1, 2.0),
                p=0.3,
            ),
            # Occlusion robustness: black fill matches mask padding
            RandomErasingAug(
                p=0.3,
                scale=(0.02, 0.15),
                ratio=(0.3, 3.3),
                value=0,
            ),
        ]
