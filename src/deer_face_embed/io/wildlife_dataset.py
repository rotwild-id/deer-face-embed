import json
from pathlib import Path
from typing import Callable, Any
import abc
import ast

import pandas as pd

from wildlife_datasets.datasets import WildlifeDataset, DownloadKaggle


class RotwildIDFaces(DownloadKaggle, WildlifeDataset, abc.ABC):
    BASE_PATH = "."
    kaggle_url = "jonaschu/rotwildid-faces"
    kaggle_type = "datasets"

    def __init__(
        self,
        root: str | None = None,
        df: pd.DataFrame | None = None,
        update_wrong_labels: bool = True,
        transform: Callable[..., Any] | None = None,
        img_load: str = "full",
        remove_unknown: bool = False,
        remove_columns: bool = False,
        check_files: bool = True,
        load_label: bool = False,
        factorize_label: bool = False,
        col_path: str = "path",
        col_label: str = "identity",
        **kwargs,
    ) -> None:
        assert root is not None
        new_root = str(Path(root) / self.BASE_PATH)
        super().__init__(
            new_root,
            df,
            update_wrong_labels,
            transform,
            img_load,
            remove_unknown,
            remove_columns,
            check_files,
            load_label,
            factorize_label,
            col_path,
            col_label,
            **kwargs,
        )
        # update summary
        with (Path(new_root) / "dataset_metadata.json").open("r") as f:
            self.summary = json.load(f)

    def create_catalogue(self) -> pd.DataFrame:
        assert self.root is not None
        df = pd.read_csv(Path(self.root) / "image_metadata.csv")
        list_columns = ["bbox", "keypoints", "segmentation"]
        for col in list_columns:
            if col not in df.columns:
                continue
            df[col] = df[col].apply(
                lambda x: list(map(float, ast.literal_eval(x))) if pd.notna(x) else None
            )
        return self.finalize_catalogue(df)


class RotwildIDFacesAffine(RotwildIDFaces):
    BASE_PATH = Path("./landmark_affine")


class RotwildIDFacesCrop(RotwildIDFaces):
    BASE_PATH = Path("./landmark_crop")


class RotwildIDFacesMask(RotwildIDFaces):
    BASE_PATH = Path("./mask")
