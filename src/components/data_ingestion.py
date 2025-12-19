import json
import os
from src.logging_system import get_logger
from src.exceptions import NLUException

logger = get_logger()


class DataIngestion:
    def __init__(self, intents_path: str, eval_path: str):
        self.intents_path = intents_path
        self.eval_path = eval_path

    def _load_json(self, file_path: str):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"Loaded data from {file_path}")
            return data

        except Exception as e:
            logger.error(f"Error loading file {file_path}")
            raise NLUException(e)

    def load_intents(self):
        logger.info("Starting intent data ingestion")
        data = self._load_json(self.intents_path)

        if "eval_data" not in data:
            raise NLUException("Invalid intents.json format: missing 'eval_data'")

        logger.info(f"Total intents loaded: {len(data['eval_data'])}")
        return data["eval_data"]

    def load_eval_data(self):
        logger.info("Starting evaluation data ingestion")
        data = self._load_json(self.eval_path)

        if "eval_data" not in data:
            raise NLUException("Invalid eval_data.json format: missing 'eval_data'")

        logger.info(f"Total evaluation samples loaded: {len(data['eval_data'])}")
        return data["eval_data"]
