import typing as T

import torch
import torch.nn as nn
from timm.layers import SelectAdaptivePool2d


from deer_face_embed.core.models.base import BaseEmbedder, BaseEmbedderConfig


# %% ResNet MODELS


class ResNetEmbedderConfig(BaseEmbedderConfig):
    KIND: T.Literal["ResNet"] = "ResNet"
    timm_model_identifier: str = "resnet50d"

    @T.override
    def load_model(
        self,
        device: torch.device = torch.device("cpu"),  # noqa: B008
    ) -> "ResNetEmbedder":
        model = ResNetEmbedder(self)
        return model.to(device)


class ResNetEmbedder(BaseEmbedder):
    ALLOWED_MODEL_IDENTIFIER: T.ClassVar[T.List[str]] = ["resnet50d"]

    def __init__(self, config: ResNetEmbedderConfig):
        super().__init__(config)
        self.config = config

    @T.override
    def get_layer_groups(self) -> dict[int, list[nn.Parameter]]:
        return {
            0: list(self.feature_extractor.layer1.parameters()),
            1: list(self.feature_extractor.layer2.parameters()),
            2: list(self.feature_extractor.layer3.parameters()),
            3: list(self.feature_extractor.layer4.parameters()),
        }

    @T.override
    def _remove_classification_head(self):
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity()
        return num_ftrs


# %% InceptionNext MODELS


class InceptionNextEmbedderConfig(BaseEmbedderConfig):
    KIND: T.Literal["inception_next"] = "inception_next"
    timm_model_identifier: str = "inception_next_base"

    @T.override
    def load_model(
        self,
        device: torch.device = torch.device("cpu"),  # noqa: B008
    ) -> "InceptionNextEmbedder":
        model = InceptionNextEmbedder(self)
        return model.to(device)


class InceptionNextEmbedder(BaseEmbedder):
    ALLOWED_MODEL_IDENTIFIER: T.ClassVar[T.List[str]] = [
        "inception_next_base",
        "inception_next_tiny",
    ]

    def __init__(self, config: InceptionNextEmbedderConfig):
        super().__init__(config)
        self.config = config

    @T.override
    def get_layer_groups(self) -> dict[int, list[nn.Parameter]]:
        return {
            i: list(stage.parameters())
            for i, stage in enumerate(self.feature_extractor.stages)
        }

    @T.override
    def _remove_classification_head(self):
        num_ftrs = self.feature_extractor.head.in_features
        self.feature_extractor.head = nn.Identity()
        return num_ftrs

    @T.override
    def _load_embedding_head(self, num_ftrs):
        embedding_head = nn.Sequential(
            SelectAdaptivePool2d(pool_type="avg", flatten=True),
            nn.Linear(num_ftrs, self.config.intermediate_size),
            nn.GELU(approximate="none"),
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(self.config.intermediate_size, self.config.embedding_size),
            nn.BatchNorm1d(self.config.embedding_size),
        )
        return embedding_head
