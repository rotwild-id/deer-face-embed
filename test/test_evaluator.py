"""Tests for MetricsCalculator — expected metric keys and perfect embedding scenario."""

import numpy as np
import pytest
import torch

from deer_face_embed.core.evaluation.evaluator import MetricsCalculator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perfect_embeddings(n_classes: int = 4, n_per_class: int = 3, dim: int = 64):
    """Build L2-normalised embeddings where each class occupies a unique unit vector.

    Within a class all embeddings are identical, so nearest-neighbour retrieval
    is perfect.  Different classes are orthogonal in the first `n_classes`
    dimensions.
    """
    vecs = []
    labels = []
    for c in range(n_classes):
        # Unit vector along axis `c`
        v = np.zeros(dim, dtype=np.float32)
        v[c] = 1.0
        for _ in range(n_per_class):
            vecs.append(v.copy())
            labels.append(c)
    return np.vstack(vecs), np.array(labels, dtype=np.int64)


def _make_random_embeddings(n: int = 20, dim: int = 64):
    """Build random L2-normalised embeddings with random integer labels."""
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(n, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    embeddings = raw / norms
    labels = rng.integers(0, 5, size=n, dtype=np.int64)
    return embeddings, labels


# ---------------------------------------------------------------------------
# MetricsCalculator — returned keys
# ---------------------------------------------------------------------------


class TestMetricsCalculatorKeys:
    EXPECTED_KEYS = {
        "AMI",
        "NMI",
        "mean_average_precision",
        "mean_reciprocal_rank",
        "precision_at_1",
    }

    def test_returns_all_expected_keys(self):
        emb, labels = _make_random_embeddings()
        metrics = MetricsCalculator.compute_metrics(
            query_embeddings=emb,
            query_labels=labels,
            reference_embeddings=emb,
            reference_labels=labels,
            query_as_part_of_ref=True,
        )
        missing = self.EXPECTED_KEYS - set(metrics.keys())
        assert not missing, f"Missing metric keys: {missing}"

    def test_all_values_are_floats(self):
        emb, labels = _make_random_embeddings()
        metrics = MetricsCalculator.compute_metrics(
            query_embeddings=emb,
            query_labels=labels,
            reference_embeddings=emb,
            reference_labels=labels,
            query_as_part_of_ref=True,
        )
        for key, value in metrics.items():
            assert isinstance(value, (float, int, np.floating)), (
                f"Metric '{key}' should be numeric, got {type(value)}"
            )

    def test_metrics_in_valid_range(self):
        """Standard accuracy metrics should lie in [0, 1]."""
        emb, labels = _make_random_embeddings()
        metrics = MetricsCalculator.compute_metrics(
            query_embeddings=emb,
            query_labels=labels,
            reference_embeddings=emb,
            reference_labels=labels,
            query_as_part_of_ref=True,
        )
        for key in ["precision_at_1", "mean_average_precision", "mean_reciprocal_rank"]:
            v = float(metrics[key])
            assert 0.0 <= v <= 1.0, f"Metric '{key}' = {v} is out of [0, 1]"


# ---------------------------------------------------------------------------
# MetricsCalculator — perfect embeddings
# ---------------------------------------------------------------------------


class TestMetricsCalculatorPerfect:
    def test_perfect_precision_at_1(self):
        """Perfect embeddings (identical within class, orthogonal across) → precision@1 = 1."""
        emb, labels = _make_perfect_embeddings(n_classes=4, n_per_class=3)
        metrics = MetricsCalculator.compute_metrics(
            query_embeddings=emb,
            query_labels=labels,
            reference_embeddings=emb,
            reference_labels=labels,
            query_as_part_of_ref=True,
        )
        assert float(metrics["precision_at_1"]) == pytest.approx(1.0, abs=1e-6), (
            f"Expected precision_at_1=1.0, got {metrics['precision_at_1']}"
        )

    def test_perfect_map(self):
        """Perfect embeddings should yield mean_average_precision ≈ 1."""
        emb, labels = _make_perfect_embeddings(n_classes=4, n_per_class=3)
        metrics = MetricsCalculator.compute_metrics(
            query_embeddings=emb,
            query_labels=labels,
            reference_embeddings=emb,
            reference_labels=labels,
            query_as_part_of_ref=True,
        )
        assert float(metrics["mean_average_precision"]) == pytest.approx(1.0, abs=1e-6), (
            f"Expected mean_average_precision=1.0, got {metrics['mean_average_precision']}"
        )

    def test_accepts_torch_tensors(self):
        """compute_metrics should accept torch.Tensor inputs as well as numpy arrays."""
        emb_np, labels_np = _make_perfect_embeddings(n_classes=3, n_per_class=4)
        emb_t = torch.from_numpy(emb_np)
        labels_t = torch.from_numpy(labels_np)
        # Should not raise
        metrics = MetricsCalculator.compute_metrics(
            query_embeddings=emb_t,
            query_labels=labels_t,
            reference_embeddings=emb_t,
            reference_labels=labels_t,
            query_as_part_of_ref=True,
        )
        assert "precision_at_1" in metrics

    def test_separate_query_and_reference(self):
        """compute_metrics with separate query / reference sets should work."""
        emb, labels = _make_perfect_embeddings(n_classes=4, n_per_class=4)
        # Use first 2 per class as query, rest as reference
        query_mask = np.zeros(len(labels), dtype=bool)
        for c in range(4):
            class_idx = np.where(labels == c)[0]
            query_mask[class_idx[:2]] = True
        query_emb = emb[query_mask]
        query_labels = labels[query_mask]
        ref_emb = emb[~query_mask]
        ref_labels = labels[~query_mask]

        metrics = MetricsCalculator.compute_metrics(
            query_embeddings=query_emb,
            query_labels=query_labels,
            reference_embeddings=ref_emb,
            reference_labels=ref_labels,
            query_as_part_of_ref=False,
        )
        assert float(metrics["precision_at_1"]) == pytest.approx(1.0, abs=1e-6)
