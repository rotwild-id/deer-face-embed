"""Base for high-level project jobs."""

import abc
import logging
import types as TS
import typing as T

import pydantic as pdt


from deer_face_embed.io import services


# TYPES

# Local job variables
Locals = T.Dict[str, T.Any]


# JOBS
class Job(abc.ABC, pdt.BaseModel):
    model_config = pdt.ConfigDict(strict=True, frozen=True, extra="forbid")
    """Base class for a job."""

    KIND: str

    mlflow_service: services.MlflowService = services.MlflowService()

    def __enter__(self) -> T.Self:
        """Enter the job context"""

        logger = logging.getLogger(__name__)

        logger.debug("[START] Mlflow service: {}", self.mlflow_service)
        self.mlflow_service.start()

        return self

    def __exit__(
        self,
        exc_type: T.Type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TS.TracebackType | None,
    ) -> T.Literal[False]:
        logger = logging.getLogger(__name__)

        logger.debug(f"[STOP] Mlflow service: {str(self.mlflow_service)}")
        self.mlflow_service.stop()
        return False  # re-raise

    @abc.abstractmethod
    def run(self) -> Locals:
        """Run the job in context.

        Returns:
            Locals: local job variables.
        """
