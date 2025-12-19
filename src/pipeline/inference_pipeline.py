from src.config_manager import load_config
from src.components.data_ingestion import DataIngestion
from src.components.prompt_builder import PromptBuilder
from src.components.llm_inference import LLMInference
from src.components.intent_classifier import IntentClassifier
from src.logging_system import get_logger
from src.exceptions import NLUException

logger = get_logger()


class InferencePipeline:
    def __init__(self):
        try:
            logger.info("Initializing Inference Pipeline")

            # Load configuration
            self.config = load_config()

            # Data ingestion
            self.data_ingestor = DataIngestion(
                intents_path=self.config["paths"]["intents_path"],
                eval_path=self.config["paths"]["eval_path"]
            )

            intents = self.data_ingestor.load_intents()

            # Prompt builder
            self.prompt_builder = PromptBuilder(intents=intents)

            # LLM inference
            llm_cfg = self.config["llm"]
            self.llm = LLMInference(
                model=llm_cfg["model"],
                temperature=llm_cfg["temperature"],
                max_tokens=llm_cfg["max_tokens"]
            )

            # Intent classifier
            self.intent_classifier = IntentClassifier(
                prompt_builder=self.prompt_builder,
                llm_inference=self.llm
            )

            logger.info("Inference Pipeline initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize inference pipeline")
            raise NLUException(e)

    def run(self, user_query: str) -> dict:
        """
        Run end-to-end intent classification
        """
        try:
            logger.info(f"Running inference for query: {user_query}")

            prediction = self.intent_classifier.predict(user_query)

            logger.info("Inference completed successfully")
            return prediction

        except Exception as e:
            logger.error("Inference pipeline execution failed")
            raise NLUException(e)
