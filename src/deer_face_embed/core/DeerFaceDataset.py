import typing as T
import random
from pathlib import Path

from torchvision.transforms.v2 import PILToTensor, Transform  # type: ignore[import-untyped]
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np


class DeerFaceDataset(Dataset):
    """
    PyTorch-style dataset for a deer face chip dataset.

    Implemented in line with https://github.com/WildlifeDatasets/wildlife-tools/blob/71aa4656d16afe4caae6d84af642bab81dc2d06d/wildlife_tools/data/dataset.py#L13


    Args:
        metadata: A pandas dataframe containing image metadata. The original index is perserved and can be used in __getitem__.
        root: Root directory of images. Provide if paths in metadata are relative. If None, absolute paths in metadata are used.
        transform: A torchvision transform that takes in an image and returns its transformed version.
        col_path: Column name in the metadata containing image file paths.
        col_label: Column name in the metadata containing class labels.
        load_label: If False, __getitem__ returns only image instead of (image, label) tuple.

    Attributes:
        labels np.array : An integers array of ordinal encoding of labels.
        labels_string np.array: A strings array of original labels.
        labels_map dict: A mapping between labels and their ordinal encoding.
        num_classes int: Return the number of unique classes in the dataset.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        root: str | None = None,
        transform: Transform | None = None,
        col_path: str = "path",
        col_label: str = "identity",
        load_label: bool = True,
        label_encoding: dict[str, int] | None = None,
    ):
        self.metadata = metadata
        self.root = Path(root) if root else None
        self.transform = transform if transform else PILToTensor()
        self.col_path = col_path
        self.col_label = col_label
        self.load_label = load_label

        if label_encoding is not None:
            identity_values = self.metadata[self.col_label].values
            labels = np.array([label_encoding[v] for v in identity_values])
            # Inverse mapping: index position → identity string
            labels_map = np.empty(len(label_encoding), dtype=object)
            for identity, idx in label_encoding.items():
                labels_map[idx] = identity
        else:
            labels, labels_map = pd.factorize(
                np.asarray(self.metadata[self.col_label].values)
            )

        self.labels = pd.Series(
            labels, index=self.metadata.index
        )  # using same index as metadata
        self.labels_map = T.cast(np.ndarray, labels_map)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data = self.metadata.loc[index]

        # load Image
        if self.root:
            img_path = self.root / data[self.col_path]
        else:
            img_path = Path(data[self.col_path])
        img = self.transform(self.load_image(img_path))

        if self.load_label:
            return img, self.labels[index]
        else:
            return img

    @staticmethod
    def load_image(path):
        img = Image.open(path).convert("RGB")
        return img

    def show_augmented_images(self, num_images=4, selected_index=None):
        """
        Display a grid of augmented images (same original) from a dataset.

        Parameters:
        - num_images (int): The number of augmented images to display in the grid. Default is 4.
        - selected_index (int, optional): The index of the image to augment. If None, a random image is selected.

        The function uses data augmentation techniques to generate multiple versions of the selected image.
        It then displays these images in a grid format using matplotlib.
        """
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid  # type: ignore[import-untyped]

        if selected_index is None:
            selected_index = self.metadata.index[random.randint(0, len(self) - 1)]

        imgs = []
        for _ in range(num_images):
            result = self[selected_index]
            imgs.append(result[0] if isinstance(result, tuple) else result)

        img_grid = make_grid(imgs, nrow=num_images)

        # Normalize the image grid to the range [0, 1]
        img_grid = (img_grid - img_grid.min()) / (img_grid.max() - img_grid.min())

        plt.imshow(
            img_grid.permute(1, 2, 0).clamp(0, 1)
        )  # Convert from Tensor image and clamp values
        plt.axis("off")
        plt.show()
