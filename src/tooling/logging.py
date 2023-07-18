import logging

from rich.logging import RichHandler
from transformers import utils


def logging_setup() -> logging.Logger:
    boto_logger = logging.getLogger("botocore.credentials")
    boto_logger.setLevel(logging.WARNING)

    utils.logging.set_verbosity_error()

    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    logger = logging.getLogger("pipeline")
    return logger
