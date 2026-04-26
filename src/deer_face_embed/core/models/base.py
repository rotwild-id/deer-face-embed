import abc
import typing as T

import pydantic as pdt
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm  # type: ignore[import-untyped]
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# %% Freeze configuration


class FreezePhaseConfig(pdt.BaseModel):
    """Defines which backbone layer groups are trainable in a training phase."""

    model_config = pdt.ConfigDict(strict=True, frozen=True, extra="forbid")
    trainable_groups: list[int] = []  # [] = only embedding head trains


class BackboneFreezeConfig(pdt.BaseModel):
    """Two-phase backbone freezing configuration.

    Controls which layer groups are trainable before and after a transition epoch.
    Each model defines its own mapping from group indices to actual parameters
    via ``get_layer_groups()``.
    """

    model_config = pdt.ConfigDict(strict=True, frozen=True, extra="forbid")
    initial: FreezePhaseConfig = FreezePhaseConfig()
    transition_epoch: int = 0  # 0 = no phase transition
    final: FreezePhaseConfig | None = None  # None = unfreeze all groups


# %% BASE


class BaseEmbedderConfig(pdt.BaseModel, abc.ABC):
    model_config = pdt.ConfigDict(strict=True, frozen=True, extra="forbid")
    KIND: str  # Base declaration that will be specialized by subclasses

    # required params
    pretrained: bool
    embedding_size: int
    timm_model_identifier: str

    # optional params
    batch_size: int = 32
    intermediate_size: int = 1024
    dropout_rate: float = 0.3
    backbone_freeze: BackboneFreezeConfig | None = None  # None = all trainable

    @pdt.model_validator(mode="after")
    def validate_freeze_pretrained(self) -> "BaseEmbedderConfig":
        if not self.pretrained and self.backbone_freeze is not None:
            raise ValueError(
                "Backbone freezing requires pretrained=True. "
                "Either set pretrained=True or remove backbone_freeze."
            )
        return self

    @abc.abstractmethod
    def load_model(
        self,
        device: torch.device = torch.device("cpu"),  # noqa: B008
    ) -> "BaseEmbedder":
        pass


class BaseEmbedder(nn.Module, abc.ABC):
    ALLOWED_MODEL_IDENTIFIER: T.ClassVar[T.List[str]]

    def __init__(
        self,
        config: BaseEmbedderConfig,
    ) -> None:
        super().__init__()

        self.config: BaseEmbedderConfig = config
        if self.ALLOWED_MODEL_IDENTIFIER is None:
            raise NotImplementedError(
                f"You must set the constant ALLOWED_MODEL_IDENTIFIER in {type(self).__name__}"
            )
        elif self.config.timm_model_identifier not in self.ALLOWED_MODEL_IDENTIFIER:
            raise ValueError(
                f"Not allowed 'timm_model_identifier'. Choose one of {', '.join(self.ALLOWED_MODEL_IDENTIFIER)}"
            )

        self.feature_extractor = self._load_feature_extractor(self.config.pretrained)
        self.fe_config = resolve_data_config({}, model=self.feature_extractor)

        # add embedding head
        num_ftrs = self._remove_classification_head()
        self.embedding_head = self._load_embedding_head(num_ftrs)

        # Apply initial freeze phase
        if self.config.backbone_freeze is not None:
            self.apply_freeze_phase(self.config.backbone_freeze.initial)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.embedding_head(x)
        x = nn.functional.normalize(x, p=2, dim=1)  # L2 normalization
        return x

    @abc.abstractmethod
    def get_layer_groups(self) -> dict[int, list[nn.Parameter]]:
        """Return a mapping from group index to list of parameters.

        Each model defines its own grouping. For example:
        - ResNet: {0: layer1, 1: layer2, 2: layer3, 3: layer4}
        - InceptionNext: {0: stages[0], ..., 3: stages[3]}
        - Swin: {0: layers.0, ..., 3: layers.3}
        - ViT: {0: blocks[0:3], 1: blocks[3:6], 2: blocks[6:9], 3: blocks[9:12]}
        """

    def apply_freeze_phase(self, phase: FreezePhaseConfig | None) -> None:
        """Apply a freeze phase configuration.

        Args:
            phase: If None, unfreeze all backbone parameters.
                Otherwise, freeze all backbone params and unfreeze only
                the specified trainable_groups.
        """
        if phase is None:
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            return

        # Freeze all backbone params first
        for param in self.feature_extractor.parameters():
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

    def generate_transform(self, no_cropping=True):
        if no_cropping:
            config = self.fe_config.copy()
            config["crop_pct"] = 1
            return create_transform(**config)
        else:
            return create_transform(**self.fe_config)

    # should be overridden most of the times
    def _remove_classification_head(self):
        num_ftrs = self.feature_extractor.head.in_features
        self.feature_extractor.head = nn.Identity()

        return num_ftrs

    def _load_feature_extractor(self, pretrained):
        model = timm.create_model(
            self.config.timm_model_identifier, pretrained=pretrained
        )
        return model

    # should be overridden most of the times
    def _load_embedding_head(self: T.Self, num_ftrs):
        embedding_head = nn.Sequential(
            nn.Linear(num_ftrs, self.config.intermediate_size),
            nn.ReLU(),
            nn.Dropout(p=self.config.dropout_rate),
            nn.Linear(self.config.intermediate_size, self.config.embedding_size),
            nn.BatchNorm1d(self.config.embedding_size),
        )
        return embedding_head

    def infer_embeddings(
        self,
        dataloader: DataLoader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # noqa: B008
    ):
        """
        Calculates the (already L2-normalized) embeddings and labels without gradient over a given data loader.
        """

        num_batches = len(dataloader)

        # no training
        self.eval()
        with torch.no_grad():
            embedding_lst = []
            labels = []

            for batch_data, batch_labels in tqdm(
                dataloader, total=num_batches, desc="Calc Embeddings"
            ):
                batch_data = batch_data.to(device)
                # Get embeddings for the current batch (L2-normalized in forward())
                embeddings = self(batch_data)
                # Move to CPU immediately to free GPU memory
                embedding_lst.append(embeddings.cpu())
                labels.extend(batch_labels)

        # Concat list of embeddings (already on CPU)
        embeddings = torch.cat(embedding_lst, dim=0)
        # convert labels to torch tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        return embeddings, labels_tensor
