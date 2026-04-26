import typing as T
import abc

import pandas as pd
import numpy as np


class BaseSplitter:
    """Base class for reproducibility and common functionality"""

    def __init__(self, identity_col="identity", exclude_identities=None, seed=42):
        self.identity_col = identity_col
        self.exclude_identities = exclude_identities or ["unknown"]
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _clean_df(self, df):
        """Helper to filter excluded identities"""
        return df[~df[self.identity_col].isin(self.exclude_identities)]

    @staticmethod
    def describe_split(
        df: pd.DataFrame,
        train_indices: T.List[int],
        test_indices: T.List[int],
        identity_column: str = "identity",
    ) -> T.Dict[str, T.Any]:
        """Generate a summary of a dataset split as a dictionary.

        Args:
            df: The full DataFrame
            train_indices: Indices of samples in the training set
            test_indices: Indices of samples in the testing set
            identity_column: Name of the column containing identity/class information

        Returns:
            A dictionary summarizing the split
        """
        # Extract dataframes for each set
        train_df = df.loc[train_indices]
        test_df = df.loc[test_indices]

        # Count samples
        n_train = len(train_indices)
        n_test = len(test_indices)
        n_unassigned = len(df) - n_train - n_test
        n_total = len(df)

        # Get identities in each set
        train_identities = set(train_df[identity_column].unique())
        test_identities = set(test_df[identity_column].unique())
        all_identities = set(df[identity_column].unique())
        unassigned_identities = all_identities - (train_identities | test_identities)

        # Count classes
        n_train_classes = len(train_identities)
        n_test_classes = len(test_identities)
        n_unassigned_classes = len(unassigned_identities)
        n_total_classes = len(all_identities)

        # Exclusive counts
        train_only_identities = train_identities - test_identities
        test_only_identities = test_identities - train_identities
        joint_identities = train_identities & test_identities

        # Count samples that belong to exclusive identities
        train_only_mask = train_df[identity_column].isin(train_only_identities)
        test_only_mask = test_df[identity_column].isin(test_only_identities)
        train_only_samples = train_only_mask.sum()
        test_only_samples = test_only_mask.sum()

        # Calculate percentages
        train_fraction = n_train / n_total * 100
        test_only_fraction = test_only_samples / n_total * 100

        # Infer split type
        if len(test_only_identities) > 0 and len(train_only_identities) > 0:
            split_type_name = "disjoint-set"
        elif len(test_only_identities) > 0:
            split_type_name = "open-set"
        else:
            split_type_name = "closed-set"

        # Create the summary dictionary
        summary = {
            "split_type": split_type_name,
            "samples": {
                "train": n_train,
                "test": n_test,
                "unassigned": n_unassigned,
                "total": n_total,
            },
            "classes": {
                "train": n_train_classes,
                "test": n_test_classes,
                "unassigned": n_unassigned_classes,
                "total": n_total_classes,
            },
            "exclusive_samples": {
                "train_only": train_only_samples,
                "test_only": test_only_samples,
            },
            "exclusive_classes": {
                "train_only": len(train_only_identities),
                "test_only": len(test_only_identities),
                "joint": len(joint_identities),
            },
            "fractions": {
                "train_fraction": train_fraction,
                "test_only_fraction": test_only_fraction,
            },
        }

        return summary

    @staticmethod
    def pretty_print_split_description(summary: T.Dict[str, T.Any]) -> str:
        """Generate a descriptive summary from a dictionary.

        Args:
            summary: A dictionary summarizing the split

        Returns:
            A formatted string describing the split
        """
        description = f"Split: {summary['split_type']}\n"
        description += f"Samples: train/test/unassigned/total = {summary['samples']['train']}/{summary['samples']['test']}/{summary['samples']['unassigned']}/{summary['samples']['total']}\n"
        description += f"Classes: train/test/unassigned/total = {summary['classes']['train']}/{summary['classes']['test']}/{summary['classes']['unassigned']}/{summary['classes']['total']}\n"
        description += f"Samples: train only/test only        = {summary['exclusive_samples']['train_only']}/{summary['exclusive_samples']['test_only']}\n"
        description += f"Classes: train only/test only/joint  = {summary['exclusive_classes']['train_only']}/{summary['exclusive_classes']['test_only']}/{summary['exclusive_classes']['joint']}\n"
        description += "\n"
        description += f"Fraction of train set     = {summary['fractions']['train_fraction']:.2f}%\n"
        description += f"Fraction of test set only = {summary['fractions']['test_only_fraction']:.2f}%\n"

        return description

    @abc.abstractmethod
    def split(self, df: pd.DataFrame) -> T.Tuple[np.ndarray, np.ndarray]:
        pass


class ClosedSetSplitter(BaseSplitter):
    """Split where all identities appear in both train/test sets"""

    def __init__(self, ratio_train=0.8, **kwargs):
        super().__init__(**kwargs)
        self.ratio_train = ratio_train

    @T.override
    def split(self, df):
        df_clean = self._clean_df(df)
        train_idx, test_idx = [], []

        for _, group in df_clean.groupby(self.identity_col):
            idx = group.index.to_numpy()
            self.rng.shuffle(idx)

            n_individual = len(idx)
            n_train = int(round(self.ratio_train * n_individual))

            # Ensure at least one sample in test if possible
            if n_train == n_individual and n_individual > 1:
                n_train -= 1
            if n_train < 1:
                n_train = 1

            train_idx.extend(idx[:n_train])
            test_idx.extend(idx[n_train:])

        return (np.array(train_idx), np.array(test_idx))


class OpenSetSplitter(BaseSplitter):
    """Split where some identities appear only in test set"""

    def __init__(
        self,
        ratio_train=0.8,
        test_size: float | None = None,
        n_test_identities: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if (test_size is None) == (n_test_identities is None):
            raise ValueError("Specify either test_size or n_test_identities")

        self.ratio_train = ratio_train
        self.test_size = test_size
        self.n_test_identities = n_test_identities

    @T.override
    def split(self, df):
        df_clean = self._clean_df(df)
        identity_counts = df_clean[self.identity_col].value_counts()
        identities = identity_counts.index.to_numpy()

        if self.n_test_identities is not None:
            selected = self.rng.choice(
                identities, self.n_test_identities, replace=False
            )
        else:
            assert self.test_size is not None
            target = int(self.test_size * len(df_clean))
            selected, cumsum = [], 0
            for identity in self.rng.permutation(identities):
                selected.append(identity)
                cumsum += identity_counts[identity]
                if cumsum >= target:
                    break

        # Split into open/closed parts
        closed_mask = ~df_clean[self.identity_col].isin(selected)
        closed_df = df_clean[closed_mask]

        # Calculate target sizes based on original ratio
        total_samples = len(df_clean)
        target_train_samples = int(self.ratio_train * total_samples)
        target_test_samples = total_samples - target_train_samples

        # Calculate how many samples are in the open set
        open_set_samples = sum(~closed_mask)

        # Calculate how many samples we need in train/test to maintain ratio_train
        total_samples = len(df_clean)
        target_train_samples = int(self.ratio_train * total_samples)
        target_test_samples = total_samples - target_train_samples

        # Calculate how many more test samples we need from closed set
        needed_closed_test = max(0, target_test_samples - open_set_samples)

        # Calculate the ratio for the closed set
        closed_set_samples = sum(closed_mask)
        closed_train_ratio = (
            closed_set_samples - needed_closed_test
        ) / closed_set_samples

        # Apply closed-set split with corrected ratio
        closed_splitter = ClosedSetSplitter(
            ratio_train=closed_train_ratio,
            identity_col=self.identity_col,
            exclude_identities=self.exclude_identities,
            seed=self.seed,
        )
        train_closed, test_closed = closed_splitter.split(closed_df)

        # Combine with held-out identities
        test_open = df_clean[~closed_mask].index.to_numpy()
        full_train = train_closed
        full_test = np.concatenate([test_closed, test_open])

        return (full_train, full_test)


class DisjointSplitter(BaseSplitter):
    """Split where identities are completely disjoint between train/test"""

    def __init__(self, test_size=None, n_test_identities=None, **kwargs):
        super().__init__(**kwargs)
        if (test_size is None) == (n_test_identities is None):
            raise ValueError("Specify either test_size or n_test_identities")

        self.test_size = test_size
        self.n_test_identities = n_test_identities

    @T.override
    def split(self, df):
        df_clean = self._clean_df(df)
        identity_counts = df_clean[self.identity_col].value_counts()
        identities = identity_counts.index.to_numpy()

        if self.n_test_identities is not None:
            selected = self.rng.choice(
                identities, self.n_test_identities, replace=False
            )
        else:
            assert self.test_size is not None
            target = int(self.test_size * len(df_clean))
            selected, cumsum = [], 0
            for identity in self.rng.permutation(identities):
                selected.append(identity)
                cumsum += identity_counts[identity]
                if cumsum >= target:
                    break

        train_mask = df_clean[self.identity_col].isin(selected)
        return (
            df_clean[~train_mask].index.to_numpy(),
            df_clean[train_mask].index.to_numpy(),
        )
