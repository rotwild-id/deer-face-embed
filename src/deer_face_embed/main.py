# src/deer_face_embed/__init__.py
import logging
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import pydantic as pdt
from deer_face_embed.jobs import JobKind


# Your existing Pydantic model stays nearly unchanged.
class MainSettings(pdt.BaseModel):
    job: JobKind = pdt.Field(..., discriminator="KIND")


config_path = str(Path(__file__).parent.joinpath("../../config").resolve())


@hydra.main(version_base=None, config_path=config_path, config_name="default")
def main(cfg):
    # Convert OmegaConf config to a plain dictionary.
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    logger = logging.getLogger(__name__)
    logger.debug(config_dict)
    # Validate and convert to your Pydantic model.
    settings = MainSettings.model_validate(config_dict)

    # Run the job in context (same as before).
    with settings.job as runner:
        runner.run()


if __name__ == "__main__":
    main()
