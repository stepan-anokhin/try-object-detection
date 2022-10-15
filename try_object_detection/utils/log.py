import logging


def configure_logging():
    """Default logging configuration."""
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
