import yaml
from src.logging_system import get_logger
from src.exceptions import NLUException

logger = get_logger()


def load_config(config_path: str = "config/config.yaml") -> dict:
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error("Failed to load config")
        raise NLUException(e)
