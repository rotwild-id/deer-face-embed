from deer_face_embed.core.models.CNNBasedModels import (
    ResNetEmbedderConfig,
    InceptionNextEmbedderConfig,
)

from deer_face_embed.core.models.VitBasedModels import (
    MegaDescriptorEmbedderConfig,
    SwinFaceEmbedderConfig,
    VitBaseFaceEmbedderConfig,
    VitDinoFaceEmbedderConfig,
    DenseVitFaceEmbedderConfig,
)


EmbedderConfigKind = (
    ResNetEmbedderConfig
    | InceptionNextEmbedderConfig
    | MegaDescriptorEmbedderConfig
    | SwinFaceEmbedderConfig
    | VitBaseFaceEmbedderConfig
    | VitDinoFaceEmbedderConfig
    | DenseVitFaceEmbedderConfig
)
