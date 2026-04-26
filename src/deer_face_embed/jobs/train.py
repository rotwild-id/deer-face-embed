import typing as T
import logging
import subprocess

import pydantic as pdt
import mlflow
from torch.utils.data import DataLoader, Subset
import numpy as np


from deer_face_embed.core.DeerFaceDataset import DeerFaceDataset
from deer_face_embed.core.evaluation.evaluator import (
    ClosedSetEvaluator,
    OpenSetEvaluator,
    VisualEvaluator,
)
from deer_face_embed.io import services
from deer_face_embed.jobs import base
from deer_face_embed.core.models import EmbedderConfigKind
from deer_face_embed.core.trainer import TrainingConfig, Trainer
from deer_face_embed.core.splitter import DisjointSplitter, ClosedSetSplitter
from deer_face_embed.io.data import LocalDatasetLoader, DatasetLoaderKind


class SplittingConfig(pdt.BaseModel):
    random_state: int = 42
    ratio_train: float = 0.75
    # how much of train set is use to calc validation metrics
    ratio_val: float = 0.1


class TrainingJob(base.Job):
    """Train and register a single AI/ML model."""

    KIND: T.Literal["TrainingJob"] = "TrainingJob"

    # splitting parameters
    splitting_config: SplittingConfig = SplittingConfig()

    # ai model parameters
    embedder_config: EmbedderConfigKind = pdt.Field(..., discriminator="KIND")

    # training parameters
    training_config: TrainingConfig

    # mlflow RUN config
    run_config: services.MlflowService.RunConfig

    # default LocalDatasetLoader
    dataset_loader: DatasetLoaderKind = pdt.Field(
        LocalDatasetLoader(), discriminator="KIND"
    )

    @T.override
    def run(self) -> base.Locals:
        logger = logging.getLogger(__name__)

        # - mlflow
        client = self.mlflow_service.client()
        logger.info(f"With client: {client.tracking_uri}")
        with self.mlflow_service.run_context(run_config=self.run_config) as run:
            logger.info(f"With run context: {run.info}")

            # Prepare and load data
            logger.info("Loading dataset")
            self.dataset_loader.prepare_data()
            df_images_metadata = self.dataset_loader.get_images_metadata()
            logger.debug("Dataset loaded")
            # logging Dataset
            logger.info("Log lineage: dataset")
            mlflow.log_input(dataset=self.dataset_loader.lineage(name="Dataset"))
            logger.debug("- Inputs lineage: {}", df_images_metadata.to_dict())

            # prepare dataset
            testset_splitter = DisjointSplitter(
                test_size=(1 - self.splitting_config.ratio_train),
                seed=self.splitting_config.random_state,
            )
            train_idx, test_idx = testset_splitter.split(df_images_metadata)
            valset_splitter = ClosedSetSplitter(
                ratio_train=(1 - self.splitting_config.ratio_val),
                seed=self.splitting_config.random_state,
            )
            train_idx, val_idx = valset_splitter.split(
                df_images_metadata.loc[train_idx, :]
            )

            # log dataset and splitting stats
            mlflow.log_params(
                {
                    "dataset.face_chip_method": self.dataset_loader.face_chip_method,
                    "dataset.n_individuals": df_images_metadata["identity"].nunique(),
                    "dataset.n_faces": len(df_images_metadata),
                    "dataset.quality_filter": ",".join(
                        self.dataset_loader.image_quality_label_included
                    ),
                    "dataset.train.n_individuals": df_images_metadata.loc[
                        train_idx, "identity"
                    ].nunique(),
                    "dataset.train.n_faces": len(train_idx),
                    "dataset.val.n_individuals": df_images_metadata.loc[
                        val_idx, "identity"
                    ].nunique(),
                    "dataset.val.n_faces": len(val_idx),
                    "dataset.test.n_individuals": df_images_metadata.loc[
                        test_idx, "identity"
                    ].nunique(),
                    "dataset.test.n_faces": len(test_idx),
                }
            )

            # initialize trainer
            trainer = Trainer(
                self.training_config,
                self.embedder_config,
                train_indices=train_idx,
                val_indices=val_idx,
                dataset_metadata=df_images_metadata,
                images_base_path=str(self.dataset_loader.data_dir),
            )

            # log git metadata
            try:
                git_branch = (
                    subprocess.check_output(
                        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                        stderr=subprocess.DEVNULL,
                    )
                    .strip()
                    .decode()
                )
                git_commit = (
                    subprocess.check_output(
                        ["git", "log", "-1", "--pretty=%B"],
                        stderr=subprocess.DEVNULL,
                    )
                    .strip()
                    .decode()
                )
                git_hash = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        stderr=subprocess.DEVNULL,
                    )
                    .strip()
                    .decode()
                )
                mlflow.set_tags(
                    {
                        "git.branch": git_branch,
                        "git.commit_message": git_commit,
                        "git.commit_hash": git_hash,
                    }
                )
            except subprocess.CalledProcessError:
                pass

            # log training and model configurations
            mlflow.log_params(self.training_config.model_dump())
            mlflow.log_params(self.embedder_config.model_dump())

            # start training
            trainer.run_train_loop()

            # log model to mlflow
            mlflow.pytorch.log_model(
                pytorch_model=trainer.model,
                artifact_path="model",
            )

            # EVALUATION
            logger.info("Running Evaluation of Training")

            dataset = DeerFaceDataset(
                metadata=df_images_metadata,
                root=str(self.dataset_loader.data_dir),
                transform=trainer.model.generate_transform(),  #  no augmentation
            )
            val_loader = DataLoader(
                Subset(dataset, val_idx),
                batch_size=self.embedder_config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            closed_set_loader = DataLoader(
                Subset(dataset, np.concatenate([train_idx, val_idx])),
                batch_size=self.embedder_config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )
            open_set_loader = DataLoader(
                Subset(dataset, test_idx),
                batch_size=self.embedder_config.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
            )

            # calc Embeddings
            val_embeddings, val_labels = trainer.model.infer_embeddings(val_loader)
            closed_set_embeddings, closed_set_labels = trainer.model.infer_embeddings(
                closed_set_loader
            )
            open_set_embeddings, open_set_labels = trainer.model.infer_embeddings(
                open_set_loader
            )

            # Evalutors
            val_set_evaluator = ClosedSetEvaluator(
                embeddings=val_embeddings, labels=val_labels
            )

            closed_set_evaluator = ClosedSetEvaluator(
                embeddings=closed_set_embeddings, labels=closed_set_labels
            )

            open_set_evaluator = OpenSetEvaluator(
                known_embeddings=closed_set_embeddings,
                known_labels=closed_set_labels,
                test_embeddings=open_set_embeddings,
                test_labels=open_set_labels,
            )

            # Run evaluations

            # Evaluation plots

            # Plot embeddings, show differences between train and validation set
            descriptive_labels = [
                dataset.labels_map[label_idx] for label_idx in closed_set_labels
            ]
            descriptive_dataset_part_labels = [
                *(len(train_idx) * ["TRAIN"]),
                *(len(val_idx) * ["VAL"]),
            ]
            title = "tSNE Embeddings labels"
            fig = VisualEvaluator.plot_embeddings(
                closed_set_embeddings.numpy(),
                color_labels=descriptive_labels,
                style_labels=descriptive_dataset_part_labels,
                title=title,
            )
            mlflow.log_figure(fig, "tSNE Embeddings labels.png")

            # Plot Roc
            (fpr, tpr), roc_auc = val_set_evaluator.calc_roc()
            roc_curve_plot = VisualEvaluator.plot_roc_curve(fpr, tpr, roc_auc)

            mlflow.log_metric("ROC-AUC", roc_auc)
            mlflow.log_figure(roc_curve_plot, artifact_file="roc.png")

            # calc and log fixed error rates
            fixed_errorrate_thresholds = closed_set_evaluator.run_threshold_evaluation()

            # Openset Evaluation: Evaluate if a unknown individual will marked as new
            open_set_evaluator.run_evaluation(fixed_errorrate_thresholds)

        return {}
