"""Tests for DeerFaceDataset — label encoding and basic interface."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from deer_face_embed.core.DeerFaceDataset import DeerFaceDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_dir_with_images():
    """Create a temp directory with dummy images for 4 identities."""
    tmp_dir = tempfile.mkdtemp()
    filenames = [f"img_{i}.jpg" for i in range(8)]
    for name in filenames:
        Image.new("RGB", (64, 64), color=(100, 100, 100)).save(Path(tmp_dir) / name)
    yield tmp_dir
    shutil.rmtree(tmp_dir)


@pytest.fixture
def full_metadata():
    """Metadata with 4 identities, 2 images each."""
    return pd.DataFrame(
        {
            "path": [f"img_{i}.jpg" for i in range(8)],
            "identity": [
                "deer_A",
                "deer_A",
                "deer_B",
                "deer_B",
                "deer_C",
                "deer_C",
                "deer_D",
                "deer_D",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Label encoding tests (moved from test_label_encoding.py)
# ---------------------------------------------------------------------------


class TestLabelEncodingConsistency:
    def test_independent_factorize_diverges(
        self, full_metadata, dataset_dir_with_images
    ):
        """Without shared encoding, splits with different identities get different mappings."""
        train_meta = full_metadata.iloc[:6].reset_index(drop=True)  # A, B, C
        val_meta = full_metadata.iloc[[0, 1, 4, 5]].reset_index(
            drop=True
        )  # A, C (no B)

        train_ds = DeerFaceDataset(metadata=train_meta, root=dataset_dir_with_images)
        val_ds = DeerFaceDataset(metadata=val_meta, root=dataset_dir_with_images)

        # pd.factorize assigns indices in order of appearance
        # train: A→0, B→1, C→2
        # val:   A→0, C→1  (C is now 1 instead of 2!)
        train_c_label = train_ds.labels[
            train_meta[train_meta["identity"] == "deer_C"].index[0]
        ]
        val_c_label = val_ds.labels[
            val_meta[val_meta["identity"] == "deer_C"].index[0]
        ]
        assert train_c_label != val_c_label, (
            "Expected divergent labels without shared encoding"
        )

    def test_shared_encoding_consistent(
        self, full_metadata, dataset_dir_with_images
    ):
        """With shared encoding, the same identity always gets the same label."""
        all_identities = sorted(full_metadata["identity"].unique())
        label_encoding = {
            identity: idx for idx, identity in enumerate(all_identities)
        }

        train_meta = full_metadata.iloc[:6].reset_index(drop=True)  # A, B, C
        val_meta = full_metadata.iloc[[0, 1, 4, 5]].reset_index(
            drop=True
        )  # A, C (no B)

        train_ds = DeerFaceDataset(
            metadata=train_meta,
            root=dataset_dir_with_images,
            label_encoding=label_encoding,
        )
        val_ds = DeerFaceDataset(
            metadata=val_meta,
            root=dataset_dir_with_images,
            label_encoding=label_encoding,
        )

        # Same identity must map to the same integer in both datasets
        for identity in ["deer_A", "deer_C"]:
            train_idx = train_meta[train_meta["identity"] == identity].index[0]
            val_idx = val_meta[val_meta["identity"] == identity].index[0]
            assert train_ds.labels[train_idx] == val_ds.labels[val_idx], (
                f"{identity} has different labels: "
                f"train={train_ds.labels[train_idx]}, val={val_ds.labels[val_idx]}"
            )

    def test_labels_map_inverse_is_correct(
        self, full_metadata, dataset_dir_with_images
    ):
        """labels_map[label_int] should return the correct identity string."""
        label_encoding = {"deer_A": 0, "deer_B": 1, "deer_C": 2, "deer_D": 3}

        ds = DeerFaceDataset(
            metadata=full_metadata,
            root=dataset_dir_with_images,
            label_encoding=label_encoding,
        )

        for identity, expected_idx in label_encoding.items():
            assert ds.labels_map[expected_idx] == identity


# ---------------------------------------------------------------------------
# Basic interface tests
# ---------------------------------------------------------------------------


class TestDeerFaceDataset:
    def test_length(self, full_metadata, dataset_dir_with_images):
        """Dataset length matches number of rows in metadata."""
        ds = DeerFaceDataset(metadata=full_metadata, root=dataset_dir_with_images)
        assert len(ds) == len(full_metadata)

    def test_getitem_returns_tuple(self, full_metadata, dataset_dir_with_images):
        """__getitem__ returns a (image, label) tuple when load_label=True."""
        ds = DeerFaceDataset(metadata=full_metadata, root=dataset_dir_with_images)
        item = ds[full_metadata.index[0]]
        assert isinstance(item, tuple), "Expected (image, label) tuple"
        assert len(item) == 2, "Tuple should have exactly 2 elements"
        img, label = item
        # Image tensor should be 3-channel
        assert img.shape[0] == 3, "Image should have 3 channels"
        # Label should be a scalar integer
        assert isinstance(int(label), int), "Label should be convertible to int"

    def test_getitem_no_label(self, full_metadata, dataset_dir_with_images):
        """__getitem__ returns only the image when load_label=False."""
        ds = DeerFaceDataset(
            metadata=full_metadata,
            root=dataset_dir_with_images,
            load_label=False,
        )
        item = ds[full_metadata.index[0]]
        assert not isinstance(item, tuple), "Expected image only (not a tuple)"

    def test_num_classes(self, full_metadata, dataset_dir_with_images):
        """num_classes property is not present; labels_map length gives class count."""
        ds = DeerFaceDataset(metadata=full_metadata, root=dataset_dir_with_images)
        num_unique = len(full_metadata["identity"].unique())
        # labels_map holds one entry per unique identity
        assert len(ds.labels_map) == num_unique

    def test_labels_are_integers(self, full_metadata, dataset_dir_with_images):
        """All label values should be non-negative integers."""
        ds = DeerFaceDataset(metadata=full_metadata, root=dataset_dir_with_images)
        for label_val in ds.labels:
            assert isinstance(int(label_val), int)
            assert int(label_val) >= 0
