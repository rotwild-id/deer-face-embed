#!/usr/bin/env python3
import typing as T

import numpy as np
import mlflow
from sklearn.manifold import TSNE  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    pairwise_distances,
    roc_curve,
    auc,
)
import torch
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator  # type: ignore[import-untyped]
import seaborn as sns
import matplotlib.pyplot as plt


class EnsureMethodExecuted:
    def __init__(self, method_name, bool_name):
        self.method_name = method_name
        self.bool_name = bool_name

    def __call__(self, method):
        def wrapper(instance, *args, **kwargs):
            # Get the method to be executed from the instance
            method_to_call = getattr(instance, self.method_name)
            # Call the method if it hasn't been called yet
            if not getattr(instance, self.bool_name):
                method_to_call()
            return method(instance, *args, **kwargs)

        return wrapper


class VisualEvaluator:
    @staticmethod
    def plot_roc_curve(fpr, tpr, roc_auc):
        fig = plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (auc-value = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        return fig

    @staticmethod
    def plot_embeddings(
        embeddings: np.ndarray,
        color_labels: T.List[str],
        style_labels: T.List[str] | None = None,
        title: str = "tSNE Embeddings",
    ) -> plt.Figure:
        tsne = TSNE().fit_transform(embeddings)

        fig, ax = plt.subplots(constrained_layout=True)
        sns.scatterplot(
            x=tsne[:, 0],
            y=tsne[:, 1],
            hue=color_labels,
            style=style_labels,
            ax=ax,
        )
        plt.title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small", ncol=2)

        return fig


class MetricsCalculator:
    _accuracy_calculator = AccuracyCalculator(
        include=(
            "AMI",
            "NMI",
            "mean_average_precision",
            "mean_reciprocal_rank",
            "precision_at_1",
        ),
        exclude=(),
        return_per_class=False,
        k=None,
    )

    @classmethod
    def compute_metrics(
        cls,
        query_embeddings: torch.Tensor | np.ndarray,
        query_labels: torch.Tensor | np.ndarray,
        reference_embeddings: torch.Tensor | np.ndarray,
        reference_labels: torch.Tensor | np.ndarray,
        query_as_part_of_ref: bool = False,
    ) -> T.Dict[str, float]:
        """Compute accuracy metrics by matching query embeddings against reference embeddings.

        This method evaluates how well embedding vectors preserve class relationships by measuring
        the accuracy of nearest-neighbor matching between query and reference sets.

        Args:
            embeddings (torch.Tensor | np.ndarray): 2D tensor or array of shape (N, D) containing
                vector embeddings for N samples, where D is the embedding dimension.
            labels (torch.Tensor | np.ndarray): 1D tensor or array of shape (N) containing
                class labels for each sample.
            query_indices (Sequence): Indices of samples to be used as queries for retrieval.
                Each index must be valid for both 'embeddings' and 'labels'.
            reference_indices (Sequence): Indices of samples to be used as the reference set
                for matching with queries. Each index must be valid for both 'embeddings' and 'labels'.
            query_as_part_of_ref (bool, optional): When True, query samples are also included
                in the reference set (self-matches are automatically ignored). Defaults to False.

        Returns:
            Dict[str, float]: A dictionary mapping metric names to their calculated values:
                - AMI (Adjusted Mutual Information): Measures clustering similarity between 0-1,
                adjusted for chance (1.0 indicates perfect clustering).
                - NMI (Normalized Mutual Information): Similar to AMI but normalized differently,
                ranges from 0-1 with 1.0 being perfect alignment.
                - mean_average_precision: The mean of average precision scores for each class,
                ranging from 0-1 (higher is better).
                - mean_reciprocal_rank: Average of 1/rank for first correct retrieval,
                ranging from 0-1 (higher is better).
                - precision_at_1: Proportion of queries where the top match is correct,
                ranging from 0-1 (higher is better).
        """

        # Select reference, add query in front of reference (requirment of AccuracyCalculator)
        if query_as_part_of_ref:
            reference_embeddings = np.concatenate(
                [query_embeddings, reference_embeddings]
            )
            reference_labels = np.concatenate([query_labels, reference_labels])

        ref_includes_query = query_as_part_of_ref or (
            bool(np.array_equal(query_labels, reference_labels))
            if (query_labels.shape == reference_labels.shape)
            else False
        )

        # For each query sample, nearest neighbors are retrieved and accuracy is computed
        # reference: This is where nearest neighbors are retrieved from.
        return cls._accuracy_calculator.get_accuracy(
            query=query_embeddings,
            query_labels=query_labels,
            reference=reference_embeddings,
            reference_labels=reference_labels,
            ref_includes_query=ref_includes_query,
        )


class ClosedSetEvaluator:
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        self.dist_matrix = None
        self.same_mask = None
        self.thresholds = None

        # bool, if high computations are done
        self._distances_computed = False
        self._error_rates_computed = False

    def _compute_distances(self):
        # Compute pairwise distances
        self.dist_matrix = pairwise_distances(self.embeddings, metric="euclidean")
        # Create arrays of "same deer?" vs. "different deer?"
        self.same_mask = np.equal.outer(self.labels, self.labels)

        self._distances_computed = True

    @EnsureMethodExecuted("_compute_distances", "_distances_computed")
    def _compute_error_rates(self):
        assert self.dist_matrix is not None
        assert self.same_mask is not None
        # Create a masking array that excludes the diagonal
        mask_no_diag = ~np.eye(len(self.labels), dtype=bool)

        # Flatten arrays but omit diagonal
        distances = self.dist_matrix[mask_no_diag]
        same_or_not = self.same_mask[mask_no_diag]

        # Sweep thresholds to find FAR/FRR
        self.thresholds = np.linspace(distances.min(), distances.max(), 100)
        FAR_list, FRR_list = [], []

        for th in self.thresholds:
            # "Accept" if dist ≤ th
            accepted_pairs = distances <= th

            # False Acceptance Rate (FAR) : fraction of impostor pairs accepted
            # Same as false positiv
            impostor_mask = ~same_or_not
            FAR = np.sum(accepted_pairs & impostor_mask) / np.sum(impostor_mask)

            # False Rejection Rate (FRR): fraction of genuine pairs rejected
            # Same as false negativ
            genuine_mask = same_or_not
            FRR = np.sum(~accepted_pairs & genuine_mask) / np.sum(genuine_mask)

            FAR_list.append(FAR)
            FRR_list.append(FRR)

        self.FAR_list = FAR_list
        self.FRR_list = FRR_list

        # flag error rates as executed
        self._error_rates_computed = True

    def fit(self):
        self._compute_distances()
        self._compute_error_rates()

    @EnsureMethodExecuted("_compute_distances", "_distances_computed")
    def calc_roc(self):
        assert self.dist_matrix is not None
        assert self.same_mask is not None
        # Exclude diagonal for ROC
        mask_no_diag = ~np.eye(len(self.labels), dtype=bool)

        # Flatten arrays but omit diagonal
        distances = self.dist_matrix[mask_no_diag]
        same_or_not = self.same_mask[mask_no_diag]

        # Compute ROC Curve and AUC
        # Use negative distances for similarity
        fpr, tpr, _ = roc_curve(same_or_not, -distances)
        roc_auc = auc(fpr, tpr)

        roc_rates = (fpr, tpr)
        return roc_rates, roc_auc

    @EnsureMethodExecuted("_compute_error_rates", "_error_rates_computed")
    def get_err_threshold(self):
        assert self.thresholds is not None
        # Find the threshold that minimizes the sum of FAR and FRR: Equal Error Rate (EER)
        EER_index = np.argmin(np.abs(np.array(self.FAR_list) - np.array(self.FRR_list)))
        EER_threshold = float(self.thresholds[EER_index])
        return EER_threshold, EER_index

    @EnsureMethodExecuted("_compute_error_rates", "_error_rates_computed")
    def get_threshold_by_far(self, target_far_value=0.1):
        assert self.thresholds is not None
        # returns the threshold which allows a False Acceptance Rate (FAR) to x % (10 % by default)
        target_index = np.argmin(np.abs(np.array(self.FAR_list) - target_far_value))
        threshold = float(self.thresholds[target_index])
        return threshold, target_index

    @EnsureMethodExecuted("_compute_error_rates", "_error_rates_computed")
    def get_threshold_by_frr(self, target_frr_value=0.1):
        assert self.thresholds is not None
        # returns the threshold which allows a False Rejection Rate (FRR) to x % (10 % by default)
        target_index = np.argmin(np.abs(np.array(self.FRR_list) - target_frr_value))
        threshold = self.thresholds[target_index]
        return threshold, target_index

    def run_threshold_evaluation(self):
        EER_threshold, EER_index = self.get_err_threshold()
        EER_FAR = self.FAR_list[EER_index]
        EER_FRR = self.FRR_list[EER_index]

        mlflow.log_metrics(
            {
                "EER_threshold": EER_threshold,
                "EER_FAR": EER_FAR,
                "EER_FRR": EER_FRR,
            }
        )

        # Threshold calculation: False Acceptance Rate at 10 %
        FAR_10_pct_threshold, FAR_10_pct_index = self.get_threshold_by_far(
            target_far_value=0.1
        )
        FAR_10_pct_FAR = self.FAR_list[FAR_10_pct_index]
        FAR_10_pct_FRR = self.FRR_list[FAR_10_pct_index]

        mlflow.log_metrics(
            {
                "FAR_10_pct_threshold": FAR_10_pct_threshold,
                "FAR_10_pct_FAR": FAR_10_pct_FAR,
                "FAR_10_pct_FRR": FAR_10_pct_FRR,
            }
        )

        thresholds = {
            "ERR": EER_threshold,
            "FAR_10_pct": FAR_10_pct_threshold,
        }

        return thresholds


class OpenSetEvaluator:
    def __init__(self, known_embeddings, known_labels, test_embeddings, test_labels):
        self.known_embeddings = (
            known_embeddings.numpy()
            if type(known_embeddings) is torch.Tensor
            else known_embeddings
        )
        self.known_labels = (
            known_labels.numpy() if type(known_labels) is torch.Tensor else known_labels
        )
        self.test_embeddings = (
            test_embeddings.numpy()
            if type(test_embeddings) is torch.Tensor
            else test_embeddings
        )
        self.test_labels = (
            test_labels.numpy() if type(test_labels) is torch.Tensor else test_labels
        )

    @staticmethod
    def compute_centroids(embeddings, labels):
        unique_ids = np.unique(labels)
        centroids = []
        for uid in unique_ids:
            centroids.append(np.mean(embeddings[labels == uid], axis=0))
        return np.vstack(centroids)

    def evaluate_new_recognition(self, threshold):
        centroids = self.compute_centroids(self.known_embeddings, self.known_labels)
        dist_matrix = pairwise_distances(
            self.test_embeddings, centroids, metric="euclidean"
        )

        total_new = len(self.test_embeddings)
        # how often is no distance closer than threshold?
        correct_new = sum(np.min(dists) > threshold for dists in dist_matrix)
        incorrect_known = total_new - correct_new

        metrics = {
            "true_unknown_rate": correct_new / total_new,
            "false_known_rate": incorrect_known / total_new,
        }

        return metrics

    def run_evaluation(self, similarity_thresholds):
        for threshold_name, threshold_value in similarity_thresholds.items():
            openset_metrics = self.evaluate_new_recognition(threshold_value)
            openset_metrics = {
                f"{threshold_name}.{metric}": openset_metrics[metric]
                for metric in openset_metrics.keys()
            }
        mlflow.log_metrics(openset_metrics)
