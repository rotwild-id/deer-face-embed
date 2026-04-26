import typing as T

import timm  # type: ignore[import-untyped]
import torch
import torch.nn as nn
from timm.layers import SelectAdaptivePool2d

from deer_face_embed.core.models.base import (
    BaseEmbedder,
    BaseEmbedderConfig,
    FreezePhaseConfig,
)


# %% Swin MODELS


class SwinFaceEmbedderConfig(BaseEmbedderConfig):
    KIND: T.Literal["Swin"] = "Swin"
    timm_model_identifier: str = "swin_base_patch4_window7_224"

    @T.override
    def load_model(
        self,
        device: torch.device = torch.device("cpu"),  # noqa: B008
    ) -> "SwinFaceEmbedder":
        model = SwinFaceEmbedder(self)
        return model.to(device)


class SwinFaceEmbedder(BaseEmbedder):
    ALLOWED_MODEL_IDENTIFIER: T.ClassVar[T.List[str]] = ["swin_base_patch4_window7_224"]

    def __init__(self, config: SwinFaceEmbedderConfig):
        super().__init__(config)
        self.config = config

    @T.override
    def get_layer_groups(self) -> dict[int, list[nn.Parameter]]:
        groups: dict[int, list[nn.Parameter]] = {}
        for name, param in self.feature_extractor.named_parameters():
            for i in range(4):
                if f"layers.{i}" in name:
                    groups.setdefault(i, []).append(param)
                    break
        return groups

    @T.override
    def _remove_classification_head(self):
        num_ftrs = self.feature_extractor.head.in_features
        self.feature_extractor.head = nn.Identity()
        return num_ftrs

    @T.override
    def _load_embedding_head(self, num_ftrs):
        embedding_head = nn.Sequential(
            SelectAdaptivePool2d(pool_type="avg", flatten=True, input_fmt="NHWC"),
            nn.Linear(num_ftrs, self.config.intermediate_size),
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(self.config.intermediate_size, self.config.embedding_size),
            nn.BatchNorm1d(self.config.embedding_size),
        )
        return embedding_head


# %% MegaDescriptor MODELS


class MegaDescriptorEmbedderConfig(BaseEmbedderConfig):
    KIND: T.Literal["MegaDescriptor"] = "MegaDescriptor"
    timm_model_identifier: str = "hf-hub:BVRA/MegaDescriptor-L-224"

    @T.override
    def load_model(
        self,
        device: torch.device = torch.device("cpu"),  # noqa: B008
    ) -> "MegaDescriptorEmbedder":
        model = MegaDescriptorEmbedder(self)
        return model.to(device)


class MegaDescriptorEmbedder(BaseEmbedder):
    ALLOWED_MODEL_IDENTIFIER: T.ClassVar[T.List[str]] = [
        "hf-hub:BVRA/MegaDescriptor-L-224"
    ]

    def __init__(self, config: MegaDescriptorEmbedderConfig):
        super().__init__(config)
        self.config = config

    @T.override
    def get_layer_groups(self) -> dict[int, list[nn.Parameter]]:
        groups: dict[int, list[nn.Parameter]] = {}
        for name, param in self.feature_extractor.named_parameters():
            for i in range(4):
                if f"layers.{i}" in name:
                    groups.setdefault(i, []).append(param)
                    break
        return groups

    @T.override
    def _remove_classification_head(self):
        num_ftrs = self.feature_extractor.head.in_features
        return num_ftrs

    @T.override
    def _load_embedding_head(self, num_ftrs):
        embedding_head = nn.Identity()
        return embedding_head


# %% VitDino MODELS


class VitDinoFaceEmbedderConfig(BaseEmbedderConfig):
    KIND: T.Literal["VitDino"] = "VitDino"
    timm_model_identifier: str = "vit_base_patch16_224.dino"

    @T.override
    def load_model(
        self,
        device: torch.device = torch.device("cpu"),  # noqa: B008
    ) -> "VitDinoFaceEmbedder":
        model = VitDinoFaceEmbedder(self)
        return model.to(device)


class VitDinoFaceEmbedder(BaseEmbedder):
    ALLOWED_MODEL_IDENTIFIER: T.ClassVar[T.List[str]] = ["vit_base_patch16_224.dino"]

    def __init__(self, config: VitDinoFaceEmbedderConfig):
        super().__init__(config)
        self.config = config

    @T.override
    def get_layer_groups(self) -> dict[int, list[nn.Parameter]]:
        groups: dict[int, list[nn.Parameter]] = {}
        for i in range(12):
            group_idx = i // 3  # 0,1,2,3
            groups.setdefault(group_idx, []).extend(
                self.feature_extractor.blocks[i].parameters()
            )
        # Include norm layer in the last group
        groups[3].extend(self.feature_extractor.norm.parameters())
        return groups

    @T.override
    def _remove_classification_head(self):
        return self.feature_extractor.num_features

    @T.override
    def _load_embedding_head(self, num_ftrs):
        intermediate_size = 512
        embedding_head = nn.Sequential(
            nn.Linear(num_ftrs, intermediate_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(intermediate_size, self.config.embedding_size),
            nn.BatchNorm1d(self.config.embedding_size),
        )
        return embedding_head


# %% VitBase MODELS


class VitBaseFaceEmbedderConfig(BaseEmbedderConfig):
    KIND: T.Literal["VitBase"] = "VitBase"
    timm_model_identifier: str = "vit_base_patch16_224"

    @T.override
    def load_model(
        self,
        device: torch.device = torch.device("cpu"),  # noqa: B008
    ) -> "VitBaseFaceEmbedder":
        model = VitBaseFaceEmbedder(self)
        return model.to(device)


class VitBaseFaceEmbedder(BaseEmbedder):
    ALLOWED_MODEL_IDENTIFIER: T.ClassVar[T.List[str]] = ["vit_base_patch16_224"]

    def __init__(self, config: VitBaseFaceEmbedderConfig):
        super().__init__(config)
        self.config = config

    @T.override
    def get_layer_groups(self) -> dict[int, list[nn.Parameter]]:
        groups: dict[int, list[nn.Parameter]] = {}
        for i in range(12):
            group_idx = i // 3  # 0,1,2,3
            groups.setdefault(group_idx, []).extend(
                self.feature_extractor.blocks[i].parameters()
            )
        # Include norm layer in the last group
        groups[3].extend(self.feature_extractor.norm.parameters())
        return groups

    @T.override
    def _load_embedding_head(self, num_ftrs):
        intermediate_size = 512
        embedding_head = nn.Sequential(
            nn.Linear(num_ftrs, intermediate_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(intermediate_size, self.config.embedding_size),
            nn.BatchNorm1d(self.config.embedding_size),
        )
        return embedding_head


class DenseVitFaceEmbedderConfig(BaseEmbedderConfig):
    KIND: T.Literal["DenseVit"] = "DenseVit"

    # Keep timm_model_identifier from BaseEmbedderConfig
    timm_model_identifier: str = "densenet121"
    vit_model: str = "vit_base_patch16_224"
    feature_fusion_dim: int = 256

    @T.override
    def load_model(
        self,
        device: torch.device = torch.device("cpu"),  # noqa: B008
    ) -> "DenseVitFaceEmbedder":  # noqa: B008
        model = DenseVitFaceEmbedder(self)
        return model.to(device)


class DenseVitFaceEmbedder(BaseEmbedder):
    # Implementation based on https://doi.org/10.1016/j.ecoinf.2023.102334
    ALLOWED_MODEL_IDENTIFIER: T.ClassVar[T.List[str]] = ["densenet121"]

    def __init__(self, config: DenseVitFaceEmbedderConfig):
        super().__init__(config)
        self.config: DenseVitFaceEmbedderConfig = config

        # Initialize Vision Transformer separately
        self.transformer = self._load_transformer()
        self.fusion_attention = nn.Sequential(
            nn.Linear(768, self.config.feature_fusion_dim), nn.Sigmoid()
        )

        # Feature fusion layers
        self.fusion_conv = nn.Conv2d(
            in_channels=1024,  # DenseNet121 final features
            out_channels=config.feature_fusion_dim,
            kernel_size=1,
        )

        # Re-apply freeze now that transformer is available
        if self.config.backbone_freeze is not None:
            self.apply_freeze_phase(self.config.backbone_freeze.initial)

    @T.override
    def _load_feature_extractor(self, pretrained: bool) -> nn.Module:
        """Override feature extractor loading to use DenseNet"""
        densenet = timm.create_model(
            self.config.timm_model_identifier,
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,),  # Get features from last dense block
        )
        return densenet

    def _load_transformer(self) -> T.Any:
        """Load Vision Transformer component"""
        transformer = timm.create_model(
            self.config.vit_model,
            pretrained=self.config.pretrained,
            num_classes=0,  # Remove classification head
        )
        return transformer

    @T.override
    def _remove_classification_head(self) -> int:
        """Returns feature dimension for embedding head"""
        return self.config.feature_fusion_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Module 1: DenseNet Feature Extraction
        dense_features = self.feature_extractor(x)[0]  # Get last feature map
        dense_features = self.fusion_conv(dense_features)

        # Module 2: Vision Transformer Processing
        vit_features = self.transformer(x)  # Shape: [B, 768]
        attention_mask = self.fusion_attention(vit_features)
        attention_mask = attention_mask.view(x.size(0), -1, 1, 1)

        # Feature Fusion
        fused_features = dense_features * attention_mask

        # Global Average Pooling
        pooled_features = torch.mean(fused_features, dim=[2, 3])

        # Generate embeddings through embedding head
        embeddings = self.embedding_head(pooled_features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    @T.override
    def get_layer_groups(self) -> dict[int, list[nn.Parameter]]:
        groups: dict[int, list[nn.Parameter]] = {}
        # Groups 0-3: DenseNet dense blocks
        for name, param in self.feature_extractor.named_parameters():
            for i, block_name in enumerate(
                ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]
            ):
                if block_name in name:
                    groups.setdefault(i, []).append(param)
                    break
        # Groups 4-7: ViT transformer blocks (3 blocks per group)
        for i in range(12):
            group_idx = 4 + (i // 3)
            groups.setdefault(group_idx, []).extend(
                self.transformer.blocks[i].parameters()
            )
        return groups

    @T.override
    def apply_freeze_phase(self, phase: FreezePhaseConfig | None) -> None:
        """Override to handle both feature_extractor and transformer backbones."""
        # Skip if transformer hasn't been initialized yet (called from super().__init__).
        # Freeze will be applied after transformer init in __init__.
        if not hasattr(self, "transformer"):
            return

        if phase is None:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            for param in self.transformer.parameters():
                param.requires_grad = True
            return

        # Freeze both backbones
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False

        # Unfreeze specified groups
        layer_groups = self.get_layer_groups()
        for group_idx in phase.trainable_groups:
            if group_idx not in layer_groups:
                raise ValueError(
                    f"Group {group_idx} not found. "
                    f"Available groups: {sorted(layer_groups.keys())}"
                )
            for param in layer_groups[group_idx]:
                param.requires_grad = True
