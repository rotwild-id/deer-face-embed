"""Tests for dataset splitter classes."""

import numpy as np
import pandas as pd
import pytest

from deer_face_embed.core.splitter import (
    ClosedSetSplitter,
    DisjointSplitter,
    OpenSetSplitter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def deer_df():
    """DataFrame with 6 identities, 4 images each (24 rows total)."""
    identities = [f"deer_{i}" for i in range(6)]
    rows = []
    for identity in identities:
        for j in range(4):
            rows.append({"path": f"{identity}/img_{j}.jpg", "identity": identity})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ClosedSetSplitter
# ---------------------------------------------------------------------------


class TestClosedSetSplitter:
    def test_all_identities_in_both_splits(self, deer_df):
        """All identities must appear in both train and test sets."""
        splitter = ClosedSetSplitter(ratio_train=0.8, seed=42)
        train_idx, test_idx = splitter.split(deer_df)

        train_identities = set(deer_df.loc[train_idx, "identity"].unique())
        test_identities = set(deer_df.loc[test_idx, "identity"].unique())
        all_identities = set(deer_df["identity"].unique())

        assert train_identities == all_identities, (
            "All identities should appear in train set (closed-set)"
        )
        assert test_identities == all_identities, (
            "All identities should appear in test set (closed-set)"
        )

    def test_approximate_split_ratio(self, deer_df):
        """Train fraction should be close to the specified ratio_train."""
        ratio_train = 0.75
        splitter = ClosedSetSplitter(ratio_train=ratio_train, seed=42)
        train_idx, test_idx = splitter.split(deer_df)

        actual_ratio = len(train_idx) / (len(train_idx) + len(test_idx))
        assert abs(actual_ratio - ratio_train) < 0.15, (
            f"Train ratio {actual_ratio:.2f} too far from target {ratio_train}"
        )

    def test_no_overlap(self, deer_df):
        """Train and test index sets must be disjoint."""
        splitter = ClosedSetSplitter(seed=42)
        train_idx, test_idx = splitter.split(deer_df)
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping indices"

    def test_covers_all_rows(self, deer_df):
        """Combined indices should cover all rows (no samples dropped)."""
        splitter = ClosedSetSplitter(seed=42)
        train_idx, test_idx = splitter.split(deer_df)
        total = len(train_idx) + len(test_idx)
        assert total == len(deer_df), (
            f"Expected {len(deer_df)} total samples, got {total}"
        )

    def test_reproducible_with_same_seed(self, deer_df):
        """Same seed should produce identical splits."""
        s1 = ClosedSetSplitter(seed=7)
        s2 = ClosedSetSplitter(seed=7)
        train1, test1 = s1.split(deer_df)
        train2, test2 = s2.split(deer_df)
        np.testing.assert_array_equal(sorted(train1), sorted(train2))
        np.testing.assert_array_equal(sorted(test1), sorted(test2))


# ---------------------------------------------------------------------------
# DisjointSplitter
# ---------------------------------------------------------------------------


class TestDisjointSplitter:
    def test_no_identity_overlap(self, deer_df):
        """Train and test must have completely disjoint identity sets."""
        splitter = DisjointSplitter(n_test_identities=2, seed=42)
        train_idx, test_idx = splitter.split(deer_df)

        train_identities = set(deer_df.loc[train_idx, "identity"].unique())
        test_identities = set(deer_df.loc[test_idx, "identity"].unique())

        overlap = train_identities & test_identities
        assert len(overlap) == 0, (
            f"Disjoint split has identity overlap: {overlap}"
        )

    def test_correct_number_of_test_identities(self, deer_df):
        """n_test_identities controls how many identities go to test."""
        n_test = 3
        splitter = DisjointSplitter(n_test_identities=n_test, seed=42)
        train_idx, test_idx = splitter.split(deer_df)

        # In DisjointSplitter, 'selected' identities become the TEST set
        test_identities = set(deer_df.loc[test_idx, "identity"].unique())
        assert len(test_identities) == n_test, (
            f"Expected {n_test} test identities, got {len(test_identities)}"
        )

    def test_test_size_fraction(self, deer_df):
        """test_size=0.5 should put roughly half the samples in test."""
        splitter = DisjointSplitter(test_size=0.5, seed=42)
        train_idx, test_idx = splitter.split(deer_df)
        total = len(train_idx) + len(test_idx)
        test_fraction = len(test_idx) / total
        # Given discrete identities with equal sizes, allow some tolerance
        assert 0.2 < test_fraction < 0.8, (
            f"Test fraction {test_fraction:.2f} is out of expected range"
        )

    def test_requires_exactly_one_size_param(self):
        """Providing both or neither size params should raise ValueError."""
        with pytest.raises(ValueError):
            DisjointSplitter()  # neither
        with pytest.raises(ValueError):
            DisjointSplitter(test_size=0.5, n_test_identities=2)  # both

    def test_excludes_unknown_identity(self):
        """Rows with the default excluded identity ('unknown') are dropped."""
        df = pd.DataFrame(
            {
                "path": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
                "identity": ["deer_A", "deer_B", "unknown", "deer_A"],
            }
        )
        splitter = DisjointSplitter(n_test_identities=1, seed=42)
        train_idx, test_idx = splitter.split(df)
        # 'unknown' rows should not appear in either split
        all_idx = set(train_idx) | set(test_idx)
        unknown_idx = set(df[df["identity"] == "unknown"].index)
        assert len(all_idx & unknown_idx) == 0, (
            "'unknown' rows should be excluded from splits"
        )


# ---------------------------------------------------------------------------
# OpenSetSplitter
# ---------------------------------------------------------------------------


class TestOpenSetSplitter:
    def test_some_identities_only_in_test(self, deer_df):
        """Open-set split should have identities exclusive to test."""
        splitter = OpenSetSplitter(n_test_identities=2, seed=42)
        train_idx, test_idx = splitter.split(deer_df)

        train_identities = set(deer_df.loc[train_idx, "identity"].unique())
        test_identities = set(deer_df.loc[test_idx, "identity"].unique())
        test_only = test_identities - train_identities
        assert len(test_only) > 0, (
            "Open-set split must have identities exclusive to the test set"
        )

    def test_requires_exactly_one_size_param(self):
        """Providing both or neither size params should raise ValueError."""
        with pytest.raises(ValueError):
            OpenSetSplitter()  # neither
        with pytest.raises(ValueError):
            OpenSetSplitter(test_size=0.5, n_test_identities=2)  # both
