"""Read/Write datasets from/to external sources/destinations."""

# %% IMPORTS

import abc
import json
import logging
from pathlib import Path
import typing as T

from mlflow.data import pandas_dataset
import pandas as pd
import pydantic as pdt

# %% CONFIGURE LOGGING

# Create a logger for this module
logger = logging.getLogger(__name__)

# %% CONSTANTS
DATASET_META_INFORMATION_FILENAME = "dataset_metadata.json"
IMAGES_METADATA_FILENAME = "image_metadata.csv"


# %% TYPINGS

Lineage: T.TypeAlias = pandas_dataset.PandasDataset


# %% READERS


class DatasetLoader(abc.ABC, pdt.BaseModel):
    """Base class for a RotwildIDFaces dataset loader.

    Use a reader to load a dataset in memory.
    e.g., to read file, database, cloud storage, ...

    """

    model_config = pdt.ConfigDict(strict=True, frozen=True, extra="forbid")

    KIND: str

    image_quality_column: str = "image_quality"
    image_quality_label_included: T.List[str] = [
        "excellent",
        "good",
        "ok",
        "poor",
    ]

    face_chip_method: T.Literal["mask", "landmark_affine", "landmark_crop"] = "mask"
    dataset_location: str = "./data/extracted"

    @property
    def data_dir(self) -> Path:
        """Directory containing metadata and images for the selected face chip method."""
        return Path(self.dataset_location) / self.face_chip_method

    @abc.abstractmethod
    def get_dataset_meta_information(self) -> dict[str, T.Any]:
        pass

    @abc.abstractmethod
    def prepare_data(self) -> None:
        """Prepares and validates data."""

    def get_images_metadata(self) -> pd.DataFrame:
        """Reads the metadata dataframe for a WildlifeDataset. Filters on given quality categories.

        Returns:
            pd.DataFrame: Dataframe with resetted index, where each row is a metadata representation of an image sample.
        """
        image_metadata = pd.read_csv(self.data_dir / IMAGES_METADATA_FILENAME)
        image_quality_mask = image_metadata[self.image_quality_column].isin(
            self.image_quality_label_included
        )
        return image_metadata[image_quality_mask].reset_index(drop=True)

    @abc.abstractmethod
    def lineage(
        self,
        name: str,
        targets: str | None = None,
        predictions: str | None = None,
    ) -> Lineage:
        """Generate lineage information.

        Args:
            name (str): dataset name.
            data (pd.DataFrame): reader dataframe.
            targets (str | None): name of the target column.
            predictions (str | None): name of the prediction column.

        Returns:
            Lineage: lineage information.
        """

    def _validate_data(self) -> None:
        if not self.data_dir.exists():
            raise ValueError(f"Dataset at {str(self.data_dir.absolute())} not found")

        df_images_meta = self.get_images_metadata()
        image_paths = df_images_meta["path"].to_list()
        for image_path_str in image_paths:
            image_path = self.data_dir / image_path_str
            if not image_path.exists() or not image_path.is_file():
                raise ValueError(f"Image file missing at: {str(image_path.absolute())}")


class LocalDatasetLoader(DatasetLoader):
    """Loads a dataset from a local directory.

    Args:
        image_quality_column (str, optional): name of column containing the image quality category of sample. Defaults to `image_quality`.
        image_quality_label_included (List[str], optional): list of image categories which should be included in loading.
        dataset_location (str, optional): Local path where the dataset (meta data and images) is located after preparation. Defaults to `data/extracted`.
    """

    KIND: T.Literal["LocalDatasetLoader"] = "LocalDatasetLoader"

    @T.override
    def prepare_data(self) -> None:
        # Local Dataset must only be validated, and not prepared.
        self._validate_data()

    @T.override
    def get_dataset_meta_information(self) -> dict[str, T.Any]:
        with (self.data_dir / DATASET_META_INFORMATION_FILENAME).open() as f:
            dataset_metainformation = json.load(f)
        return dataset_metainformation

    @T.override
    def lineage(
        self,
        name: str,
        targets: str | None = None,
        predictions: str | None = None,
    ) -> Lineage:
        dataset_created_at = self.get_dataset_meta_information().get(
            "created", "unknown"
        )
        source_str = f"LocalDataset({str(self.data_dir.absolute())}; \
            CreatedAt: {dataset_created_at}); \
            Filtered on: {', '.join(self.image_quality_label_included)}"

        return pandas_dataset.from_pandas(
            df=self.get_images_metadata(),
            name=name,
            source=source_str,
            targets=targets,
            predictions=predictions,
        )


class KaggleDatasetLoader(DatasetLoader):
    """Loads the RotwildIDFaces dataset from Kaggle using WildlifeDataset classes.

    Downloads and caches the dataset automatically. Requires KAGGLE_API_TOKEN environment variable.

    Args:
        face_chip_method (str): Which face chip variant to load.
            - "mask": mask-based extraction
            - "landmark_affine": affine-aligned chips
            - "landmark_crop": bounding-box crop chips
        dataset_location (str, optional): Local directory to download/cache the dataset.
            Defaults to "./data/kaggle".
    """

    KIND: T.Literal["KaggleDatasetLoader"] = "KaggleDatasetLoader"
    dataset_location: str = "./data/kaggle"

    def _cls_map(self):
        from deer_face_embed.io.wildlife_dataset import (
            RotwildIDFacesAffine,
            RotwildIDFacesCrop,
            RotwildIDFacesMask,
        )

        return {
            "mask": RotwildIDFacesMask,
            "landmark_affine": RotwildIDFacesAffine,
            "landmark_crop": RotwildIDFacesCrop,
        }

    @T.override
    def prepare_data(self) -> None:
        """Download the Kaggle dataset if not already cached."""
        dataset_cls = self._cls_map()[self.face_chip_method]
        dataset_cls.get_data(self.dataset_location)

    @T.override
    def get_dataset_meta_information(self) -> dict[str, T.Any]:
        with (self.data_dir / DATASET_META_INFORMATION_FILENAME).open() as f:
            dataset_metainformation = json.load(f)
        return dataset_metainformation

    def load(self):
        """Instantiate and return the WildlifeDataset for the configured face_chip_method.

        Returns:
            RotwildIDFaces: The loaded WildlifeDataset instance.
        """
        dataset_cls = self._cls_map()[self.face_chip_method]
        return dataset_cls(root=self.dataset_location)

    @T.override
    def lineage(
        self,
        name: str,
        targets: str | None = None,
        predictions: str | None = None,
    ) -> Lineage:
        from deer_face_embed.io.wildlife_dataset import RotwildIDFaces

        dataset_info = self.get_dataset_meta_information()
        kaggle_url = f"https://www.kaggle.com/datasets/{RotwildIDFaces.kaggle_url}"
        created_at = dataset_info.get("created", "unknown")
        source_str = (
            f"KaggleDataset({kaggle_url}; "
            f"method={self.face_chip_method}; "
            f"CreatedAt: {created_at}); "
            f"Filtered on: {', '.join(self.image_quality_label_included)}"
        )

        return pandas_dataset.from_pandas(
            df=self.get_images_metadata(),
            name=name,
            source=source_str,
            targets=targets,
            predictions=predictions,
        )


DatasetLoaderKind = LocalDatasetLoader | KaggleDatasetLoader
