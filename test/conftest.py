"""Shared pytest fixtures for the face embedding model test suite."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image


@pytest.fixture
def dummy_rgb_image(size=(224, 224)):
    """Return a random RGB PIL Image."""
    arr = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


@pytest.fixture
def dummy_image_tensor():
    """Return a (3, 224, 224) float tensor on CPU."""
    return torch.rand(3, 224, 224, device="cpu")


@pytest.fixture
def dummy_batch_tensor(batch_size=4):
    """Return a (4, 3, 224, 224) float tensor on CPU."""
    return torch.rand(batch_size, 3, 224, 224, device="cpu")


@pytest.fixture
def sample_metadata_df():
    """Return a small pandas DataFrame with columns ['path', 'identity']."""
    return pd.DataFrame(
        {
            "path": [
                "deer_A/img_0.jpg",
                "deer_A/img_1.jpg",
                "deer_B/img_0.jpg",
                "deer_B/img_1.jpg",
                "deer_C/img_0.jpg",
                "deer_C/img_1.jpg",
            ],
            "identity": ["deer_A", "deer_A", "deer_B", "deer_B", "deer_C", "deer_C"],
        }
    )
